import json
import logging
import os
from typing import Any

import pandas as pd
import torch
import wandb
from peft import LoraConfig
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from Callbacks import WandBPredictionLogger
from Dataset import ChestXrayDataset
from finetune_config import (
    EVAL_STEPS,
    EVAL_STRATEGY,
    EXPERIMENT_TAG,
    GRADIENT_ACCUMULATION_STEPS,
    HUB_PRIVATE_REPO,
    LEARNING_RATE,
    LOGGING_STEPS,
    LOG_DIR,
    LOG_PATH,
    LORA_ALPHA,
    LORA_BIAS,
    LORA_DROPOUT,
    LORA_MODULES_TO_SAVE,
    LORA_R,
    LORA_TARGET_MODULES,
    LORA_TASK_TYPE,
    MODE,
    MODEL_ID,
    MODELS_DIR,
    NUM_TRAIN_EPOCHS,
    OUTPUT_DIR,
    OUTPUT_RUN_CONFIG_PATH,
    PER_DEVICE_EVAL_BATCH_SIZE,
    PER_DEVICE_TRAIN_BATCH_SIZE,
    PROMPT_PATH,
    PUSH_TO_HUB,
    RESUME_FROM_CHECKPOINT,
    RUN_CONFIG_PATH,
    RUN_NAME,
    SAMPLE_RANDOM_SEED,
    SAVE_STRATEGY,
    TRAIN_CSV_PATH,
    TRAIN_SAMPLE_FRAC,
    VAL_CSV_PATH,
    VAL_SAMPLE_FRAC,
    WANDB_DIR,
    WANDB_PROJECT,
    build_run_config,
)
from utils import ensure_dirs, get_latest_checkpoint, jsonable, save_run_config

# torchrun --nproc_per_node=2 src/finetune.py

local_rank = int(os.environ.get("LOCAL_RANK", "0"))
is_main_process = local_rank == 0

ensure_dirs(MODELS_DIR, OUTPUT_DIR, LOG_DIR, WANDB_DIR)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()],
)
log = logging.getLogger(f"medgemma-train-{EXPERIMENT_TAG}")

run_config = build_run_config(local_rank=local_rank, is_main_process=is_main_process)
save_run_config(run_config, RUN_CONFIG_PATH, OUTPUT_RUN_CONFIG_PATH)

log.info("Starting MedGemma fine-tuning script")
log.info("Run config:\n%s", json.dumps(jsonable(run_config), indent=2))

# ------------------------------------------------------------
# Load dataframes
# ------------------------------------------------------------
log.info("Loading training and validation CSVs")
train_df = pd.read_csv(TRAIN_CSV_PATH)
val_df = pd.read_csv(VAL_CSV_PATH)

train_df = train_df.sample(frac=TRAIN_SAMPLE_FRAC, random_state=SAMPLE_RANDOM_SEED).reset_index(drop=True)
val_df = val_df.sample(frac=VAL_SAMPLE_FRAC, random_state=SAMPLE_RANDOM_SEED).reset_index(drop=True)

run_config["data"]["train_samples"] = len(train_df)
run_config["data"]["val_samples"] = len(val_df)
save_run_config(run_config, RUN_CONFIG_PATH, OUTPUT_RUN_CONFIG_PATH)

log.info("Train samples: %d", len(train_df))
log.info("Validation samples: %d", len(val_df))

# ------------------------------------------------------------
# Load prompt
# ------------------------------------------------------------
log.info("Loading instruction prompt from %s", PROMPT_PATH)
with open(PROMPT_PATH, "r", encoding="utf-8") as f:
    instruction_prompt = f.read().strip()
log.info("Instruction prompt loaded")

# ------------------------------------------------------------
# Create datasets
# ------------------------------------------------------------
log.info("Creating datasets")
train_dataset = ChestXrayDataset(train_df, instruction_prompt, mode=MODE)
val_dataset = ChestXrayDataset(val_df, instruction_prompt, mode=MODE)
log.info("Datasets created")

# ------------------------------------------------------------
# Load model and processor
# ------------------------------------------------------------
log.info("Loading model: %s", MODEL_ID)
if torch.cuda.get_device_capability()[0] < 8:
    log.error("GPU does not support bfloat16")
    raise ValueError("GPU does not support bfloat16, please use a GPU that supports bfloat16.")

model_kwargs = {
    "attn_implementation": "eager",
    "torch_dtype": torch.bfloat16,
    "quantization_config": BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_storage=torch.bfloat16,
    ),
}

model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, **model_kwargs)
processor = AutoProcessor.from_pretrained(MODEL_ID)
processor.tokenizer.padding_side = "right"
log.info("Model and processor loaded successfully")

# ------------------------------------------------------------
# LoRA configuration
# ------------------------------------------------------------
peft_config = LoraConfig(
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    r=LORA_R,
    bias=LORA_BIAS,
    target_modules=LORA_TARGET_MODULES,
    task_type=LORA_TASK_TYPE,
    modules_to_save=LORA_MODULES_TO_SAVE,
)


def collate_fn(examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    texts = []
    images = []

    for example in examples:
        images.append([img.convert("RGB") for img in example["images"]])
        texts.append(
            processor.apply_chat_template(
                example["messages"],
                add_generation_prompt=False,
                tokenize=False,
            ).strip()
        )

    batch = processor(
        text=texts,
        images=images,
        return_tensors="pt",
        padding=True,
    )

    labels = batch["input_ids"].clone()
    image_token_id = [
        processor.tokenizer.convert_tokens_to_ids(processor.tokenizer.special_tokens_map["boi_token"])
    ]

    labels[labels == processor.tokenizer.pad_token_id] = -100
    labels[labels == image_token_id] = -100
    labels[labels == 262144] = -100

    batch["labels"] = labels
    return batch


resolved_resume_checkpoint = RESUME_FROM_CHECKPOINT
if RESUME_FROM_CHECKPOINT == "auto":
    latest_checkpoint = get_latest_checkpoint(OUTPUT_DIR)
    resolved_resume_checkpoint = str(latest_checkpoint) if latest_checkpoint else None
    if resolved_resume_checkpoint:
        log.info("Auto-resume checkpoint found: %s", resolved_resume_checkpoint)
    else:
        log.info("Auto-resume requested but no checkpoint found in %s", OUTPUT_DIR)

args = SFTConfig(
    output_dir=str(OUTPUT_DIR),
    num_train_epochs=NUM_TRAIN_EPOCHS,
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    gradient_checkpointing=True,
    optim="adamw_torch_fused",
    logging_steps=LOGGING_STEPS,
    save_strategy=SAVE_STRATEGY,
    eval_strategy=EVAL_STRATEGY,
    eval_steps=EVAL_STEPS,
    learning_rate=LEARNING_RATE,
    bf16=True,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="linear",
    push_to_hub=PUSH_TO_HUB,
    hub_private_repo=HUB_PRIVATE_REPO,
    report_to=["wandb"],
    gradient_checkpointing_kwargs={"use_reentrant": False},
    ddp_find_unused_parameters=False,
    dataset_kwargs={"skip_prepare_dataset": True},
    remove_unused_columns=False,
    label_names=["labels"],
)

if is_main_process:
    wandb.init(
        project=WANDB_PROJECT,
        name=RUN_NAME,
        dir=str(WANDB_DIR),
        config=jsonable(run_config),
    )

log.info("Initializing trainer")
trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    peft_config=peft_config,
    processing_class=processor,
    data_collator=collate_fn,
)

trainer.add_callback(
    WandBPredictionLogger(
        dataset=val_dataset,
        processor=processor,
        num_samples=10,
    )
)

log.info("Starting training")
trainer.train(resume_from_checkpoint=resolved_resume_checkpoint)
log.info("Training completed")

log.info("Saving model to %s", OUTPUT_DIR)
trainer.save_model()

log.info("Cleaning up resources")
del model
del trainer
torch.cuda.empty_cache()
train_dataset.close()
val_dataset.close()

log.info("Finished MedGemma fine-tuning script")
