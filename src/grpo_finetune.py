import logging
import os
import random
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pandas as pd
import torch
import wandb
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer

from Callbacks import GRPOHeldoutRewardEvaluator, WandBGRPOPredictionLogger
from Dataset import ChestXrayGRPODataset, ChestXrayPPODataset
from Reward import RewardModel
from grpo_finetune_config import (
    ADAPTER_OUTPUT_DIR,
    ADAPTER_RUN_CONFIG_PATH,
    BASE_MODEL_ID,
    DATASET_FRACTION,
    EVAL_SAMPLES,
    EXPERIMENT_TAG,
    GRPO_BETA,
    GRPO_GRADIENT_ACCUMULATION_STEPS,
    GRPO_LEARNING_RATE,
    GRPO_LOGGING_STEPS,
    GRPO_MAX_COMPLETION_LENGTH,
    GRPO_NUM_GENERATIONS,
    GRPO_NUM_TRAIN_EPOCHS,
    GRPO_SAVE_STEPS,
    GRPO_SAVE_STRATEGY,
    GRPO_STEPS_PER_GENERATION,
    HF_REPO_NAME,
    JUDGE_DEVICE_STR,
    LOG_DIR,
    LOG_PATH,
    MODE,
    MODELS_DIR,
    OUTPUT_DIR,
    OUTPUT_RUN_CONFIG_PATH,
    POLICY_DEVICE_STR,
    PROMPT_PATH,
    RUN_CONFIG_PATH,
    RUN_NAME,
    SEED,
    SFT_CHECKPOINT,
    TRAIN_CSV_PATH,
    WANDB_DIR,
    WANDB_PROJECT,
    build_run_config,
)
from grpo_utils import messages_to_text
from utils import ensure_dirs, jsonable, save_run_config


os.environ["WANDB_PROJECT"] = WANDB_PROJECT

# ------------------------------------------------------------
# Devices
# ------------------------------------------------------------
POLICY_DEVICE = torch.device(POLICY_DEVICE_STR)
JUDGE_DEVICE = torch.device(JUDGE_DEVICE_STR)


# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ------------------------------------------------------------
# Logging setup
# ------------------------------------------------------------
ensure_dirs(MODELS_DIR, OUTPUT_DIR, ADAPTER_OUTPUT_DIR, LOG_DIR, WANDB_DIR)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()],
)
log = logging.getLogger(f"medgemma-grpo-train-{EXPERIMENT_TAG}")

run_config = build_run_config()
save_run_config(run_config, RUN_CONFIG_PATH, OUTPUT_RUN_CONFIG_PATH, ADAPTER_RUN_CONFIG_PATH)

log.info("Starting MedGemma GRPO fine-tuning script")
log.info("Run config:\n%s", jsonable(run_config))

wandb.init(
    project=WANDB_PROJECT,
    name=RUN_NAME,
    dir=str(WANDB_DIR),
    config=jsonable(run_config),
)


# ------------------------------------------------------------
# Load dataframes, prompt, make dataset
# ------------------------------------------------------------
full_train_df = pd.read_csv(TRAIN_CSV_PATH)

# Speed-focused subset (configurable)
dataset_fraction = min(1.0, max(0.0, float(DATASET_FRACTION)))
sample_count = min(len(full_train_df), max(1, int(len(full_train_df) * dataset_fraction)))
sampled_df = full_train_df.sample(n=sample_count, random_state=SEED).reset_index(drop=True)
eval_size = min(EVAL_SAMPLES, max(1, len(sampled_df) // 20))
if eval_size >= len(sampled_df):
    eval_size = max(1, len(sampled_df) - 1)

eval_idx = sampled_df.sample(n=eval_size, random_state=SEED).index
eval_df = sampled_df.loc[eval_idx].reset_index(drop=True)
train_df = sampled_df.drop(index=eval_idx).reset_index(drop=True)

with open(PROMPT_PATH, "r", encoding="utf-8") as f:
    instruction_prompt = f.read().strip()

base_train_dataset = ChestXrayPPODataset(
    train_df,
    instruction_prompt,
    mode=MODE,
)
train_dataset = ChestXrayGRPODataset(base_train_dataset, single_image_for_grpo=True)

base_eval_dataset = ChestXrayPPODataset(
    eval_df,
    instruction_prompt,
    mode=MODE,
)
eval_dataset = ChestXrayGRPODataset(base_eval_dataset, single_image_for_grpo=True)

run_config["data"] = {
    "dataset_fraction": dataset_fraction,
    "full_train_rows": len(full_train_df),
    "sampled_rows": len(sampled_df),
    "train_rows": len(train_dataset),
    "eval_rows": len(eval_dataset),
    "eval_size": eval_size,
}
save_run_config(run_config, RUN_CONFIG_PATH, OUTPUT_RUN_CONFIG_PATH, ADAPTER_RUN_CONFIG_PATH)


# ------------------------------------------------------------
# Load SFT model and processor
# ------------------------------------------------------------
model_kwargs = {
    "attn_implementation": "sdpa",
    "torch_dtype": torch.bfloat16,
    "quantization_config": BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_storage=torch.bfloat16,
    ),
}

base_model = AutoModelForImageTextToText.from_pretrained(
    BASE_MODEL_ID,
    **model_kwargs,
)

model = PeftModel.from_pretrained(
    base_model,
    str(SFT_CHECKPOINT),
    is_trainable=True,
)
model.config.use_cache = False

processor = AutoProcessor.from_pretrained(BASE_MODEL_ID)
processor.tokenizer.padding_side = "left"
if processor.tokenizer.pad_token_id is None and processor.tokenizer.eos_token_id is not None:
    processor.tokenizer.pad_token = processor.tokenizer.eos_token
if processor.tokenizer.eos_token_id is not None:
    model.generation_config.eos_token_id = processor.tokenizer.eos_token_id
if processor.tokenizer.pad_token_id is not None:
    model.generation_config.pad_token_id = processor.tokenizer.pad_token_id


# ------------------------------------------------------------
# Reward model + bridge function for GRPO
# ------------------------------------------------------------
reward_model = RewardModel(device=JUDGE_DEVICE)


def grpo_reward_fn(
    prompts: list[Any],
    completions: list[Any],
    reference_text: list[str] | None = None,
    trainer_state=None,
    **kwargs,
) -> list[float]:
    """
    Custom GRPO reward function.

    The `reference_text` list is forwarded from dataset columns by TRL.
    """

    del trainer_state, kwargs

    if reference_text is None:
        raise ValueError("Missing `reference_text` in reward function inputs.")

    prompt_texts = [messages_to_text(p) for p in prompts]
    completion_texts = [messages_to_text(c) for c in completions]

    rewards = reward_model.score(
        prompts=prompt_texts,
        generations=completion_texts,
        references=reference_text,
    )

    if len(rewards) != len(completion_texts):
        raise ValueError(
            f"Reward length mismatch: got {len(rewards)} rewards for {len(completion_texts)} completions."
        )

    return [float(r) for r in rewards]


# ------------------------------------------------------------
# GRPO configuration and trainer
# ------------------------------------------------------------
grpo_config = GRPOConfig(
    output_dir=str(OUTPUT_DIR),
    run_name=RUN_NAME,
    learning_rate=GRPO_LEARNING_RATE,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=GRPO_GRADIENT_ACCUMULATION_STEPS,
    # Must yield a generation batch divisible by num_generations.
    # With per_device_train_batch_size=1 and 1 process, this needs to be >=2 when num_generations=2.
    steps_per_generation=GRPO_STEPS_PER_GENERATION,
    num_generations=GRPO_NUM_GENERATIONS,
    max_completion_length=GRPO_MAX_COMPLETION_LENGTH,
    beta=GRPO_BETA,
    num_train_epochs=GRPO_NUM_TRAIN_EPOCHS,
    # Gemma3 currently breaks in transformers continuous batching/paged generation path.
    # Keep standard generation for stability.
    use_transformers_paged=False,
    remove_unused_columns=False,
    bf16=True,
    logging_steps=GRPO_LOGGING_STEPS,
    save_steps=GRPO_SAVE_STEPS,
    save_strategy=GRPO_SAVE_STRATEGY,
    report_to=["wandb"],
)

# Avoid torch.nn.DataParallel for VLM GRPO. DataParallel can break multimodal
# input scattering (image token / feature count mismatches). Keep policy on a
# single training GPU while still allowing a separate judge GPU.
if torch.cuda.device_count() > 1:
    grpo_config._n_gpu = 1
    log.info("Forcing GRPO trainer to single policy GPU to avoid DataParallel multimodal scatter issues.")

trainer = GRPOTrainer(
    model=model,
    reward_funcs=[grpo_reward_fn],
    args=grpo_config,
    train_dataset=train_dataset,
    processing_class=processor,
)

log.info(
    "Run preflight | seed=%d train_rows=%d eval_rows=%d beta=%.5f max_completion_length=%d "
    "steps_per_generation=%d output_dir=%s logs_dir=%s hf_repo=%s",
    SEED,
    len(train_dataset),
    len(eval_dataset),
    grpo_config.beta,
    grpo_config.max_completion_length,
    grpo_config.steps_per_generation,
    OUTPUT_DIR,
    LOG_DIR,
    f"Jerjes/{HF_REPO_NAME}",
)
wandb.config.update(
    {
        "seed": SEED,
        "train_rows": len(train_dataset),
        "eval_rows": len(eval_dataset),
        "beta": grpo_config.beta,
        "steps_per_generation": grpo_config.steps_per_generation,
        "max_completion_length": grpo_config.max_completion_length,
        "grpo_output_dir": str(OUTPUT_DIR),
        "log_dir": str(LOG_DIR),
        "hf_repo": f"Jerjes/{HF_REPO_NAME}",
    },
    allow_val_change=True,
)

trainer.add_callback(
    WandBGRPOPredictionLogger(
        dataset=train_dataset,
        processor=processor,
        reward_model=reward_model,
        num_samples=2,
        max_new_tokens=64,
        log_every_n_steps=50,
    )
)
trainer.add_callback(
    GRPOHeldoutRewardEvaluator(
        dataset=eval_dataset,
        processor=processor,
        reward_model=reward_model,
        num_samples=eval_size,
        max_new_tokens=grpo_config.max_completion_length,
        log_every_n_steps=100,
    )
)


# ------------------------------------------------------------
# Train
# ------------------------------------------------------------
log.info("Starting GRPO training")
trainer.train()


# ------------------------------------------------------------
# Save GRPO adapter
# ------------------------------------------------------------
log.info("Saving GRPO LoRA adapter to %s", ADAPTER_OUTPUT_DIR)
model.save_pretrained(ADAPTER_OUTPUT_DIR)
processor.save_pretrained(ADAPTER_OUTPUT_DIR)
model.push_to_hub(f"Jerjes/{HF_REPO_NAME}", private=True)

log.info("GRPO adapter saved successfully")


# ------------------------------------------------------------
# Save and cleanup
# ------------------------------------------------------------
log.info("Cleaning up resources")
del model
torch.cuda.empty_cache()
train_dataset.close()
eval_dataset.close()
log.info("Finished MedGemma GRPO fine-tuning script")
