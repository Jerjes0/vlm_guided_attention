import os
import wandb
import logging
import torch
import pandas as pd

from datetime import datetime
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig, TrainerCallback
from typing import Any
from peft import LoraConfig
from trl import SFTConfig


from Dataset import ChestXrayDataset
from Callbacks import WandBPredictionLogger

 # torchrun --nproc_per_node=2 finetune.py

# ------------------------------------------------------------
# Overal settings
# ------------------------------------------------------------

mode = 'image_and_heatmap'  # 'image' or 'image_and_heatmap'
date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# ------------------------------------------------------------
# Logging setup
# ------------------------------------------------------------
LOG_DIR = "../csv/loggings"
os.makedirs(LOG_DIR, exist_ok=True)

log_path = os.path.join(LOG_DIR, f"train_medgemma_{mode}_{date_str}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler(),  # still prints to stdout
    ],
)

log = logging.getLogger("medgemma-train")
log.info("Starting MedGemma fine-tuning script")
log.info(f"Mode: {mode}, date: {date_str}")

# ------------------------------------------------------------
# Load dataframes
# ------------------------------------------------------------
log.info("Loading training and validation CSVs")

train_df = pd.read_csv('../csv/train.csv')
val_df = pd.read_csv('../csv/val.csv')

log.info(f"Train samples: {len(train_df)}")
log.info(f"Validation samples: {len(val_df)}")


# ------------------------------------------------------------
# Load prompt
# ------------------------------------------------------------
log.info("Loading instruction prompt")

with open(f"../csv/prompts/{mode}.txt", "r") as f:
    instruction_prompt = f.read().strip()

log.info("Instruction prompt loaded")


# ------------------------------------------------------------
# Create Datasets
# ------------------------------------------------------------
log.info("Creating datasets")

train_dataset = ChestXrayDataset(train_df, instruction_prompt, mode=mode)
val_dataset = ChestXrayDataset(val_df, instruction_prompt, mode=mode)

log.info("Datasets created")


# ------------------------------------------------------------
# Load Model
# ------------------------------------------------------------
model_name = "medgemma-1.5-4b-it"
model_id = f"google/{model_name}"

log.info(f"Loading model: {model_id}")

if torch.cuda.get_device_capability()[0] < 8:
    log.error("GPU does not support bfloat16")
    raise ValueError("GPU does not support bfloat16, please use a GPU that supports bfloat16.")

model_kwargs = dict(
    attn_implementation="eager",
    torch_dtype=torch.bfloat16,
    # device_map="auto", #  commenting to see if I can parallelize
)

model_kwargs["quantization_config"] = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=model_kwargs["torch_dtype"],
    bnb_4bit_quant_storage=model_kwargs["torch_dtype"],
)

model = AutoModelForImageTextToText.from_pretrained(model_id, **model_kwargs)
processor = AutoProcessor.from_pretrained(model_id)

processor.tokenizer.padding_side = "right"

log.info("Model and processor loaded successfully")


# ------------------------------------------------------------
# LoRA Configuration
# ------------------------------------------------------------
log.info("Setting LoRA configuration")

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=16,
    bias="none",
    target_modules="all-linear",
    task_type="CAUSAL_LM",
    modules_to_save=[
        "lm_head",
        "embed_tokens",
    ],
)

log.info("LoRA configuration set")


# ------------------------------------------------------------
# Data Collator
# ------------------------------------------------------------
def collate_fn(examples: list[dict[str, Any]]):
    texts = []
    images = []

    for example in examples:
        images.append([img.convert("RGB") for img in example["images"]])
        texts.append(
            processor.apply_chat_template(
                example["messages"],
                add_generation_prompt=False,
                tokenize=False
            ).strip()
        )

    batch = processor(
        text=texts,
        images=images,
        return_tensors="pt",
        padding=True
    )

    labels = batch["input_ids"].clone()

    image_token_id = [
        processor.tokenizer.convert_tokens_to_ids(
            processor.tokenizer.special_tokens_map["boi_token"]
        )
    ]

    labels[labels == processor.tokenizer.pad_token_id] = -100
    labels[labels == image_token_id] = -100
    labels[labels == 262144] = -100

    batch["labels"] = labels
    return batch

# ------------------------------------------------------------
# Training Configuration
# ------------------------------------------------------------
log.info("Configuring training arguments")

num_train_epochs = 1
learning_rate = 2e-4

args = SFTConfig(
    output_dir=f"{model_name}-{mode}-lora-{date_str}",
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=2, #4
    per_device_eval_batch_size=2, #4
    gradient_accumulation_steps=8, #4
    gradient_checkpointing=True,
    optim="adamw_torch_fused",
    logging_steps=50,
    save_strategy="epoch",
    eval_strategy="steps",
    eval_steps=50,
    learning_rate=learning_rate,
    bf16=True,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="linear",
    push_to_hub=True,
    hub_private_repo=True,
    report_to=["tensorboard", "wandb"],
    gradient_checkpointing_kwargs={"use_reentrant": False},
    dataset_kwargs={"skip_prepare_dataset": True},
    remove_unused_columns=False,
    label_names=["labels"], 
)

log.info("Training arguments configured")

# ------------------------------------------------------------
# Weights & Biases Initialization
# ------------------------------------------------------------
log.info("Initializing Weights & Biases run")

wandb.init(
    project="medgemma-chest-xray",
    name=f"{model_name}-{mode}-{date_str}",
    config={
        "model": model_name,
        "num_train_epochs": num_train_epochs,
        "learning_rate": learning_rate,
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "lora_r": peft_config.r,
        "lora_alpha": peft_config.lora_alpha,
        "lora_dropout": peft_config.lora_dropout,
        "batch_size": args.per_device_train_batch_size,
        "gradient_accumulation": args.gradient_accumulation_steps,
    },
)


# ------------------------------------------------------------
# Training
# ------------------------------------------------------------
from trl import SFTTrainer


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
        num_samples=5,
    )
)


log.info("Starting training")
trainer.train()
log.info("Training completed")


# ------------------------------------------------------------
# Save and cleanup
# ------------------------------------------------------------
log.info("Saving model")
trainer.save_model()

log.info("Cleaning up resources")

del model
del trainer
torch.cuda.empty_cache()

train_dataset.close()
val_dataset.close()

log.info("Finished MedGemma fine-tuning script")
