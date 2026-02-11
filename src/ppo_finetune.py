import os
import wandb
import torch
import logging
import pandas as pd

from datetime import datetime
from typing import Any, List

from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from peft import PeftModel
from trl import PPOConfig, PPOTrainer

from Dataset import ChestXrayDataset
from Reward import compute_reward

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

log_path = os.path.join(LOG_DIR, f"train_medgemma_ppo_{mode}_{date_str}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler(),  # still prints to stdout
    ],
)

log = logging.getLogger("medgemma-ppo-train")
log.info("Starting MedGemma PPO fine-tuning script")
log.info(f"Mode: {mode}, date: {date_str}")

# ------------------------------------------------------------
# Load dataframes, prompt, make dataset
# ------------------------------------------------------------

train_df = pd.read_csv("../csv/train.csv")

with open(f"../csv/prompts/{mode}.txt", "r") as f:
    instruction_prompt = f.read().strip()

train_dataset = ChestXrayDataset(
    train_df,
    instruction_prompt,
    mode=mode,
)

# ------------------------------------------------------------
# Load SFT model and processor
# ------------------------------------------------------------

base_model_id = "google/medgemma-1.5-4b-it"
sft_checkpoint = "path/to/your/sft/lora/checkpoint"

model_kwargs = dict(
    attn_implementation="eager",
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
)

processor = AutoProcessor.from_pretrained(base_model_id)
processor.tokenizer.padding_side = "right"

# ------------------------------------------------------------
# Reference model  -- KL anchoring
# ------------------------------------------------------------

ref_model = PeftModel.from_pretrained(
    AutoModelForImageTextToText.from_pretrained(base_model_id, **model_kwargs),
    sft_checkpoint,
)

ref_model.eval()
for p in ref_model.parameters():
    p.requires_grad = False

# ------------------------------------------------------------
# Data collator (no labels)
# ------------------------------------------------------------

def collate_fn(examples: List[dict[str, Any]]):
    texts, images = [], []

    for ex in examples:
        images.append([img.convert("RGB") for img in ex["images"]])
        texts.append(
            processor.apply_chat_template(
                ex["messages"],
                add_generation_prompt=True,
                tokenize=False,
            )
        )

    batch = processor(
        text=texts,
        images=images,
        return_tensors="pt",
        padding=True,
    )

    return batch

# ------------------------------------------------------------
# PPO Configuration
# ------------------------------------------------------------

ppo_config = PPOConfig(
    learning_rate=1e-5,
    batch_size=2,
    mini_batch_size=1,
    gradient_accumulation_steps=8,
    kl_coef=0.1,
    cliprange=0.2,
    cliprange_value=0.2,
    vf_coef=0.1,
    bf16=True,
)

# Initialize PPO trainer

ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=model,
    ref_model=ref_model,
    tokenizer=processor.tokenizer,
    data_collator=collate_fn,
    train_dataset=train_dataset,
)

# ------------------------------------------------------------
# Generation and reward loop
# ------------------------------------------------------------


for step, batch in enumerate(ppo_trainer.dataloader):
    #batch = {k: v.to(model.device) for k, v in batch.items()}

    # 1. Generate
    generation_outputs = ppo_trainer.generate(
        **batch,
        max_new_tokens=128,
        do_sample=True,
        top_p=0.9,
        temperature=1.0,
    )


    responses = processor.tokenizer.batch_decode(
        generation_outputs,
        skip_special_tokens=True,
    )

    prompts = processor.tokenizer.batch_decode(
        batch["input_ids"],
        skip_special_tokens=True,
    )

    # 2. Compute reward
    rewards = compute_reward(prompts, responses)

    # 3. PPO update
    stats = ppo_trainer.step(
        batch["input_ids"],
        generation_outputs,
        rewards,
    )

    if step % 10 == 0:
        wandb.log({
            "ppo/reward_mean": sum(rewards) / len(rewards),
            **stats,
        })

# ------------------------------------------------------------
# Save PPO adapter 
# ------------------------------------------------------------

ppo_adapter_dir = f"medgemma-ppo-from-sft-lora-{mode}-{date_str}"
os.makedirs(ppo_adapter_dir, exist_ok=True)

log.info(f"Saving PPO LoRA adapter to {ppo_adapter_dir}")

model.save_pretrained(ppo_adapter_dir)
processor.save_pretrained(ppo_adapter_dir)

model.push_to_hub(f"Jerjes/{ppo_adapter_dir}", private=True)

log.info("PPO adapter saved successfully")

# ------------------------------------------------------------
# Save and cleanup
# ------------------------------------------------------------


log.info("Cleaning up resources")

del model

torch.cuda.empty_cache()

train_dataset.close()

log.info("Finished MedGemma PPO fine-tuning script")
