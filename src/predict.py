import os
import logging
import torch
import pandas as pd

from datetime import datetime
from transformers import pipeline, AutoProcessor
from tqdm import tqdm

from Dataset import ChestXrayDataset

# ------------------------------------------------------------
# Overall settings
# ------------------------------------------------------------

mode = 'image_and_heatmap'  # 'image' or 'image_and_heatmap'
state = 'zs'
model_checkpoint = "google/medgemma-1.5-4b-it" #"./medgemma-1.5-4b-it-image-lora-2026-02-06_11-01-04" 
date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
max_new_tokens = 512
num_test_samples = None  # Set to a number to test on subset, None for all

# ------------------------------------------------------------
# Logging setup
# ------------------------------------------------------------
LOG_DIR = "../csv/loggings"
os.makedirs(LOG_DIR, exist_ok=True)

log_path = os.path.join(LOG_DIR, f"predict_medgemma_{state}_{mode}_{date_str}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler(),
    ],
)

log = logging.getLogger("medgemma-predict")
log.info("Starting MedGemma prediction script")
log.info(f"State: {state}, Mode: {mode}, date: {date_str}")

# ------------------------------------------------------------
# Load dataframe
# ------------------------------------------------------------
log.info("Loading test CSV")

test_df = pd.read_csv('../csv/test.csv')

if num_test_samples is not None:
    test_df = test_df.head(num_test_samples)
    log.info(f"Using first {num_test_samples} test samples")

log.info(f"Test samples: {len(test_df)}")

# ------------------------------------------------------------
# Load prompt
# ------------------------------------------------------------
log.info("Loading instruction prompt")

with open(f"../csv/prompts/{mode}.txt", "r") as f:
    instruction_prompt = f.read().strip()

log.info("Instruction prompt loaded")

# ------------------------------------------------------------
# Create Dataset
# ------------------------------------------------------------
log.info("Creating test dataset")

test_dataset = ChestXrayDataset(test_df, instruction_prompt, mode=mode)

log.info("Test dataset created")

# ------------------------------------------------------------
# Load Model with Pipeline
# ------------------------------------------------------------
log.info(f"Loading model from checkpoint: {model_checkpoint}")

if torch.cuda.get_device_capability()[0] < 8:
    log.error("GPU does not support bfloat16")
    raise ValueError("GPU does not support bfloat16, please use a GPU that supports bfloat16.")

# Load the fine-tuned model
pt_pipe = pipeline(
    "image-text-to-text",
    model=model_checkpoint,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# Load processor
processor = AutoProcessor.from_pretrained(model_checkpoint)

# Set generation config
pt_pipe.model.generation_config.do_sample = False  # Deterministic
pt_pipe.model.generation_config.pad_token_id = processor.tokenizer.eos_token_id

log.info("Model and pipeline loaded successfully")
log.info(f"Model device: {pt_pipe.device}")

# ------------------------------------------------------------
# Generate predictions
# ------------------------------------------------------------
log.info("Generating predictions")

all_predictions = []
all_ground_truths = []

for idx in tqdm(range(len(test_dataset)), desc="Predicting"):
    # Get example
    example = test_dataset[idx]
    
    # Apply chat template (like in collator, but for inference)
    prompt_text = processor.apply_chat_template(
        example["messages"][:-1],  # System + user only (no assistant)
        add_generation_prompt=True,  # Add the assistant prompt
        tokenize=False
    ).strip()
    
    # Process inputs (like in collator)
    inputs = processor(
        text=[prompt_text],
        images=[example["images"]],
        return_tensors="pt",
        padding=True
    ).to(pt_pipe.device)
    
    # Generate
    with torch.no_grad():
        outputs = pt_pipe.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    
    # Decode
    generated = processor.decode(outputs[0], skip_special_tokens=True)
    
    # Remove prompt from output
    if generated.startswith(prompt_text):
        prediction = generated[len(prompt_text):].strip()
    else:
        prediction = generated.strip()
    
    # Get ground truth
    ground_truth = example["messages"][-1]["content"][0]["text"]
    
    all_predictions.append(prediction)
    all_ground_truths.append(ground_truth)
    
    # Log progress every 10 samples
    if (idx + 1) % 10 == 0:
        log.info(f"Processed {idx + 1}/{len(test_dataset)} samples")

log.info(f"Generated {len(all_predictions)} predictions")

# ------------------------------------------------------------
# Save predictions
# ------------------------------------------------------------
output_csv = f"../csv/outputs/predictions_{state}_{mode}_{date_str}.csv"

log.info(f"Saving predictions to: {output_csv}")

# Create output dataframe
predictions_df = test_df.copy()
predictions_df['prediction'] = all_predictions
predictions_df['ground_truth'] = all_ground_truths

predictions_df.to_csv(output_csv, index=False)

log.info(f"Predictions saved successfully to {output_csv}")

# # ------------------------------------------------------------
# # Display sample predictions
# # ------------------------------------------------------------
# log.info("\n" + "="*80)
# log.info("Sample Predictions:")
# log.info("="*80)

# num_display = min(3, len(predictions_df))
# for i in range(num_display):
#     log.info(f"\nSample {i+1}:")
#     log.info(f"Ground Truth: {predictions_df.iloc[i]['ground_truth'][:200]}...")
#     log.info(f"Prediction:   {predictions_df.iloc[i]['prediction'][:200]}...")
#     log.info("-"*80)

# ------------------------------------------------------------
# Cleanup
# ------------------------------------------------------------
log.info("Cleaning up resources")

test_dataset.close()
del pt_pipe
torch.cuda.empty_cache()

log.info("Finished MedGemma prediction script")