import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoProcessor, pipeline

from Dataset import ChestXrayDataset

# ------------------------------------------------------------
# Run configuration (edit only here)
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODE = "image_and_heatmap"  # "image" or "image_and_heatmap"
STATE = "zs"

# Can be a hub model ID or a local run path under models/.
MODEL_CHECKPOINT = "google/medgemma-1.5-4b-it"
# MODEL_CHECKPOINT = PROJECT_ROOT / "models" / "medgemma-1.5-4b-it-single_image_overlay-image_and_heatmap-lora-YYYY-MM-DD_HH-MM-SS"

MAX_NEW_TOKENS = 512
NUM_TEST_SAMPLES: int | None = None  # Set to int for subset, None for full test CSV

TEST_CSV_PATH = PROJECT_ROOT / "csv" / "test.csv"
PROMPT_PATH = PROJECT_ROOT / "csv" / "prompts" / f"{MODE}.txt"

RUN_TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
RUN_NAME = f"predict-{STATE}-{MODE}-{RUN_TIMESTAMP}"

LOG_DIR = PROJECT_ROOT / "csv" / "loggings" / "sft"
LOG_PATH = LOG_DIR / f"{RUN_NAME}.log"
RUN_CONFIG_PATH = LOG_DIR / f"run_config_{RUN_NAME}.json"

OUTPUT_DIR = PROJECT_ROOT / "csv" / "outputs"
OUTPUT_CSV = OUTPUT_DIR / f"predictions_{STATE}_{MODE}_{RUN_TIMESTAMP}.csv"
OUTPUT_RUN_CONFIG_PATH = OUTPUT_DIR / f"run_config_{RUN_NAME}.json"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def resolve_model_checkpoint(model_checkpoint: str | Path) -> str:
    if isinstance(model_checkpoint, Path):
        return str(model_checkpoint.resolve())
    return model_checkpoint


LOG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()],
)
log = logging.getLogger("medgemma-predict")

resolved_model_checkpoint = resolve_model_checkpoint(MODEL_CHECKPOINT)
run_config = {
    "run_name": RUN_NAME,
    "mode": MODE,
    "state": STATE,
    "model_checkpoint": MODEL_CHECKPOINT,
    "resolved_model_checkpoint": resolved_model_checkpoint,
    "generation": {
        "max_new_tokens": MAX_NEW_TOKENS,
        "num_test_samples": NUM_TEST_SAMPLES,
        "do_sample": False,
    },
    "paths": {
        "project_root": PROJECT_ROOT,
        "test_csv_path": TEST_CSV_PATH,
        "prompt_path": PROMPT_PATH,
        "log_dir": LOG_DIR,
        "log_path": LOG_PATH,
        "run_config_path": RUN_CONFIG_PATH,
        "output_dir": OUTPUT_DIR,
        "output_csv": OUTPUT_CSV,
    },
}
run_config_jsonable = _jsonable(run_config)


def save_run_config(config: dict[str, Any]) -> None:
    with open(RUN_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    with open(OUTPUT_RUN_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


save_run_config(run_config_jsonable)

log.info("Starting MedGemma prediction script")
log.info("Run config:\n%s", json.dumps(run_config_jsonable, indent=2))

# ------------------------------------------------------------
# Load dataframe
# ------------------------------------------------------------
log.info("Loading test CSV")
test_df = pd.read_csv(TEST_CSV_PATH)

if NUM_TEST_SAMPLES is not None:
    test_df = test_df.head(NUM_TEST_SAMPLES)
    log.info("Using first %d test samples", NUM_TEST_SAMPLES)

run_config_jsonable["data"] = {"test_samples": len(test_df)}
save_run_config(run_config_jsonable)
log.info("Test samples: %d", len(test_df))

# ------------------------------------------------------------
# Load prompt
# ------------------------------------------------------------
log.info("Loading instruction prompt from %s", PROMPT_PATH)
with open(PROMPT_PATH, "r", encoding="utf-8") as f:
    instruction_prompt = f.read().strip()
log.info("Instruction prompt loaded")

# ------------------------------------------------------------
# Create Dataset
# ------------------------------------------------------------
log.info("Creating test dataset")
test_dataset = ChestXrayDataset(test_df, instruction_prompt, mode=MODE)
log.info("Test dataset created")

# ------------------------------------------------------------
# Load Model with Pipeline
# ------------------------------------------------------------
log.info("Loading model from checkpoint: %s", resolved_model_checkpoint)

if torch.cuda.get_device_capability()[0] < 8:
    log.error("GPU does not support bfloat16")
    raise ValueError("GPU does not support bfloat16, please use a GPU that supports bfloat16.")

pt_pipe = pipeline(
    "image-text-to-text",
    model=resolved_model_checkpoint,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

processor = AutoProcessor.from_pretrained(resolved_model_checkpoint)

pt_pipe.model.generation_config.do_sample = False
pt_pipe.model.generation_config.pad_token_id = processor.tokenizer.eos_token_id

log.info("Model and pipeline loaded successfully")
log.info("Model device: %s", pt_pipe.device)

# ------------------------------------------------------------
# Generate predictions
# ------------------------------------------------------------
log.info("Generating predictions")

all_predictions = []
all_ground_truths = []

for idx in tqdm(range(len(test_dataset)), desc="Predicting"):
    example = test_dataset[idx]

    prompt_text = processor.apply_chat_template(
        example["messages"][:-1],
        add_generation_prompt=True,
        tokenize=False,
    ).strip()

    inputs = processor(
        text=[prompt_text],
        images=[example["images"]],
        return_tensors="pt",
        padding=True,
    ).to(pt_pipe.device)

    with torch.no_grad():
        outputs = pt_pipe.model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    generated = processor.decode(outputs[0], skip_special_tokens=True)

    if generated.startswith(prompt_text):
        prediction = generated[len(prompt_text):].strip()
    else:
        prediction = generated.strip()

    ground_truth = example["messages"][-1]["content"][0]["text"]

    all_predictions.append(prediction)
    all_ground_truths.append(ground_truth)

    if (idx + 1) % 10 == 0:
        log.info("Processed %d/%d samples", idx + 1, len(test_dataset))

log.info("Generated %d predictions", len(all_predictions))

# ------------------------------------------------------------
# Save predictions
# ------------------------------------------------------------
log.info("Saving predictions to: %s", OUTPUT_CSV)

predictions_df = test_df.copy()
predictions_df["prediction"] = all_predictions
predictions_df["ground_truth"] = all_ground_truths

predictions_df.to_csv(OUTPUT_CSV, index=False)

log.info("Predictions saved successfully to %s", OUTPUT_CSV)

# ------------------------------------------------------------
# Cleanup
# ------------------------------------------------------------
log.info("Cleaning up resources")

test_dataset.close()
del pt_pipe
torch.cuda.empty_cache()

log.info("Finished MedGemma prediction script")
