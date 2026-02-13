from datetime import datetime
from pathlib import Path
from typing import Any

# ------------------------------------------------------------
# Run configuration (edit only here)
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODE = "image_and_heatmap"  # "image" or "image_and_heatmap"
EXPERIMENT_TAG = "single_image_overlay"
WANDB_PROJECT = "medgemma-chest-xray-single-image-overlay-rl"

SEED = 42
EVAL_SAMPLES = 50
DATASET_FRACTION = 0.5

POLICY_DEVICE_STR = "cuda:0"
JUDGE_DEVICE_STR = "cuda:1"

BASE_MODEL_ID = "google/medgemma-1.5-4b-it"
BASE_MODEL_NAME = BASE_MODEL_ID.split("/", 1)[-1]
SFT_CHECKPOINT = (
    PROJECT_ROOT
    / "models"
    / "medgemma-1.5-4b-it-single_image_overlay-image_and_heatmap-lora-2026-02-12_11-47-10"
)

GRPO_LEARNING_RATE = 1e-5
GRPO_NUM_GENERATIONS = 4
GRPO_MAX_COMPLETION_LENGTH = 64
GRPO_GRADIENT_ACCUMULATION_STEPS = 8
GRPO_STEPS_PER_GENERATION = 8
GRPO_BETA = 0.03
GRPO_NUM_TRAIN_EPOCHS = 1.0
GRPO_LOGGING_STEPS = 10
GRPO_SAVE_STEPS = 10
GRPO_SAVE_STRATEGY = "steps"

TRAIN_CSV_PATH = PROJECT_ROOT / "csv" / "train.csv"
PROMPT_PATH = PROJECT_ROOT / "csv" / "prompts" / f"{MODE}.txt"

RUN_TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
RUN_NAME = f"{BASE_MODEL_NAME}-{EXPERIMENT_TAG}-{MODE}-grpo-{RUN_TIMESTAMP}"
HF_REPO_NAME = f"medgemma-grpo-from-sft-lora-{EXPERIMENT_TAG}-{MODE}-test-{RUN_TIMESTAMP}"

MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = MODELS_DIR / RUN_NAME
OUTPUT_RUN_CONFIG_PATH = OUTPUT_DIR / "run_config.json"

ADAPTER_OUTPUT_DIR = MODELS_DIR / HF_REPO_NAME
ADAPTER_RUN_CONFIG_PATH = ADAPTER_OUTPUT_DIR / "run_config.json"

LOG_DIR = PROJECT_ROOT / "csv" / "loggings" / "grpo"
LOG_PATH = LOG_DIR / f"train_{RUN_NAME}.log"
RUN_CONFIG_PATH = LOG_DIR / f"run_config_{RUN_NAME}.json"

WANDB_DIR = PROJECT_ROOT / "wandb"


def build_run_config() -> dict[str, Any]:
    return {
        "run_name": RUN_NAME,
        "mode": MODE,
        "experiment_tag": EXPERIMENT_TAG,
        "wandb_project": WANDB_PROJECT,
        "seed": SEED,
        "eval_samples": EVAL_SAMPLES,
        "dataset_fraction": DATASET_FRACTION,
        "devices": {
            "policy_device": POLICY_DEVICE_STR,
            "judge_device": JUDGE_DEVICE_STR,
        },
        "model": {
            "base_model_id": BASE_MODEL_ID,
            "sft_checkpoint": SFT_CHECKPOINT,
            "hf_repo_name": HF_REPO_NAME,
        },
        "training": {
            "learning_rate": GRPO_LEARNING_RATE,
            "num_generations": GRPO_NUM_GENERATIONS,
            "max_completion_length": GRPO_MAX_COMPLETION_LENGTH,
            "gradient_accumulation_steps": GRPO_GRADIENT_ACCUMULATION_STEPS,
            "steps_per_generation": GRPO_STEPS_PER_GENERATION,
            "beta": GRPO_BETA,
            "num_train_epochs": GRPO_NUM_TRAIN_EPOCHS,
            "logging_steps": GRPO_LOGGING_STEPS,
            "save_steps": GRPO_SAVE_STEPS,
            "save_strategy": GRPO_SAVE_STRATEGY,
        },
        "paths": {
            "project_root": PROJECT_ROOT,
            "train_csv": TRAIN_CSV_PATH,
            "prompt_path": PROMPT_PATH,
            "models_dir": MODELS_DIR,
            "output_dir": OUTPUT_DIR,
            "adapter_output_dir": ADAPTER_OUTPUT_DIR,
            "log_dir": LOG_DIR,
            "log_path": LOG_PATH,
            "run_config_path": RUN_CONFIG_PATH,
            "wandb_dir": WANDB_DIR,
        },
    }
