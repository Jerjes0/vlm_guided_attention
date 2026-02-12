import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import logging
from datetime import datetime
from typing import Any

import pandas as pd
import torch
import wandb

from trl import GRPOConfig, GRPOTrainer

from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel

from Dataset import ChestXrayPPODataset, ChestXrayGRPODataset
from Reward import RewardModel
from grpo_utils import messages_to_text
from Callbacks import WandBGRPOPredictionLogger


# ------------------------------------------------------------
# Overall settings
# ------------------------------------------------------------

mode = "image_and_heatmap"  # "image" or "image_and_heatmap"
experiment_tag = "single_image_overlay"
wandb_project = "medgemma-chest-xray-single-image-overlay-rl"
os.environ["WANDB_PROJECT"] = wandb_project
date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# -------------------------------------------------------------
# Divide GPUs
# --------------------------------------------------------------

POLICY_DEVICE = torch.device("cuda:0")
JUDGE_DEVICE = torch.device("cuda:1")

# ------------------------------------------------------------
# Logging setup
# ------------------------------------------------------------

LOG_DIR = f"../csv/loggings_{experiment_tag}"
os.makedirs(LOG_DIR, exist_ok=True)

log_path = os.path.join(LOG_DIR, f"train_medgemma_grpo_{experiment_tag}_{mode}_{date_str}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler(),
    ],
)

log = logging.getLogger(f"medgemma-grpo-train-{experiment_tag}")
log.info("Starting MedGemma GRPO fine-tuning script")
log.info("Mode: %s, date: %s, experiment: %s", mode, date_str, experiment_tag)

wandb.init(
    project=wandb_project,
    name=f"grpo-{experiment_tag}-{mode}-{date_str}",
    config={
        "mode": mode,
        "experiment_tag": experiment_tag,
        "learning_rate": 1e-5,
        "num_generations": 2,
        "max_completion_length": 64,
    },
)

# ------------------------------------------------------------
# Load dataframes, prompt, make dataset
# ------------------------------------------------------------

train_df = pd.read_csv("../csv/train.csv")

# Small sample for early smoke runs
train_df = train_df.sample(n=20, random_state=42).reset_index(drop=True)

with open(f"../csv/prompts/{mode}.txt", "r") as f:
    instruction_prompt = f.read().strip()

base_train_dataset = ChestXrayPPODataset(
    train_df,
    instruction_prompt,
    mode=mode,
)
train_dataset = ChestXrayGRPODataset(base_train_dataset, single_image_for_grpo=True)

# ------------------------------------------------------------
# Load SFT model and processor
# ------------------------------------------------------------

base_model_id = "google/medgemma-1.5-4b-it"
sft_checkpoint = "./medgemma-1.5-4b-it-image_and_heatmap-lora-2026-02-07_21-13-31"

model_kwargs = dict(
    attn_implementation="sdpa",
    torch_dtype=torch.bfloat16,
)

model_kwargs["quantization_config"] = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_storage=torch.bfloat16,
)

base_model = AutoModelForImageTextToText.from_pretrained(
    base_model_id,
    **model_kwargs,
)

model = PeftModel.from_pretrained(
    base_model,
    sft_checkpoint,
    is_trainable=True,
)
model.config.use_cache = False

processor = AutoProcessor.from_pretrained(base_model_id)
processor.tokenizer.padding_side = "left"

# -------------------------------------------------------------
# Reward model + bridge function for GRPO
# -------------------------------------------------------------

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
# GRPO Configuration and trainer
# ------------------------------------------------------------

grpo_output_dir = f"./medgemma-grpo-runs-{experiment_tag}-{mode}-{date_str}"

grpo_config = GRPOConfig(
    output_dir=grpo_output_dir,
    run_name=f"grpo-{experiment_tag}-{mode}-{date_str}",
    learning_rate=1e-5,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    # Must yield a generation batch divisible by num_generations.
    # With per_device_train_batch_size=1 and 1 process, this needs to be >=2 when num_generations=2.
    steps_per_generation=2,
    num_generations=2,
    max_completion_length=64,
    beta=0.0,
    num_train_epochs=1.0,
    # Gemma3 currently breaks in transformers continuous batching/paged generation path.
    # Keep standard generation for stability.
    use_transformers_paged=False,
    remove_unused_columns=False,
    bf16=True,
    logging_steps=1,
    save_steps=50,
    save_strategy="steps",
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
trainer.add_callback(
    WandBGRPOPredictionLogger(
        dataset=train_dataset,
        processor=processor,
        reward_model=reward_model,
        num_samples=4,
        max_new_tokens=64,
        log_every_n_steps=25,
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

grpo_adapter_dir = f"medgemma-grpo-from-sft-lora-{experiment_tag}-{mode}-test-{date_str}"
os.makedirs(grpo_adapter_dir, exist_ok=True)

log.info("Saving GRPO LoRA adapter to %s", grpo_adapter_dir)
model.save_pretrained(grpo_adapter_dir)
processor.save_pretrained(grpo_adapter_dir)

model.push_to_hub(f"Jerjes/{grpo_adapter_dir}", private=True)

log.info("GRPO adapter saved successfully")

# ------------------------------------------------------------
# Save and cleanup
# ------------------------------------------------------------

log.info("Cleaning up resources")

del model

torch.cuda.empty_cache()

train_dataset.close()

log.info("Finished MedGemma GRPO fine-tuning script")
