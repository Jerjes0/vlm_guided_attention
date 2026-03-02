from datetime import datetime
from pathlib import Path
from typing import Any

# ------------------------------------------------------------
# Run configuration (edit only here)
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODE = "image"  # "image" or "image_and_heatmap"
EXPERIMENT_TAG = "single_image_overlay_new"

BASE_MODEL_NAME = "medgemma-1.5-4b-it"
MODEL_ID = f"google/{BASE_MODEL_NAME}"
WANDB_PROJECT = "new-medgemma-chest-xray-single-image-overlay"

TRAIN_CSV_PATH = PROJECT_ROOT / "csv" / "train.csv"
VAL_CSV_PATH = PROJECT_ROOT / "csv" / "val.csv"
PROMPT_PATH = PROJECT_ROOT / "csv" / "prompts" / f"{MODE}.txt"

TRAIN_SAMPLE_FRAC = 1  # set to 1.0 for full training set
VAL_SAMPLE_FRAC = 1    # set to 1.0 for full validation set
SAMPLE_RANDOM_SEED = 42

NUM_TRAIN_EPOCHS = 3 #1
PER_DEVICE_TRAIN_BATCH_SIZE = 2
PER_DEVICE_EVAL_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 5e-5 #2e-4
LOGGING_STEPS = 25
EVAL_STEPS = 100 #10
SAVE_STRATEGY = "steps"
SAVE_STEPS = 100
EVAL_STRATEGY = "steps"
ENABLE_EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 3
EARLY_STOPPING_THRESHOLD = 0.0
BEST_MODEL_METRIC = "eval_loss"
BEST_MODEL_GREATER_IS_BETTER = False
PUSH_TO_HUB = True
HUB_PRIVATE_REPO = True

LORA_R = 32
LORA_ALPHA = 2 * LORA_R
LORA_DROPOUT = 0.05
LORA_BIAS = "none"
# LORA_TARGET_MODULES = "all-linear"

LORA_TARGET_MODULES = [
    # Language Model (Gemma) layers
    "q_proj", "k_proj", "v_proj", "o_proj", 
    "gate_proj", "up_proj", "down_proj",
    # Vision Tower (SigLIP) & Projector layers
    "vision_model.encoder.layers.*.self_attn.q_proj",
    "vision_model.encoder.layers.*.self_attn.k_proj",
    "vision_model.encoder.layers.*.self_attn.v_proj",
    "vision_model.encoder.layers.*.self_attn.out_proj",
    "vision_model.encoder.layers.*.mlp.fc1",
    "vision_model.encoder.layers.*.mlp.fc2",
    "multi_modal_projector.linear",
]

LORA_TASK_TYPE = "CAUSAL_LM"
LORA_MODULES_TO_SAVE = ["lm_head", "embed_tokens"]

# Resume options: None | "auto" | "/full/path/to/checkpoint"
RESUME_FROM_CHECKPOINT: str | None = None

RUN_TIMESTAMP = datetime.now().strftime("%Y-%m-%d")
RUN_NAME = f"{BASE_MODEL_NAME}-{EXPERIMENT_TAG}-{MODE}-lora-{RUN_TIMESTAMP}"

MODELS_DIR = PROJECT_ROOT / "models" / "heatmap_analysis"
OUTPUT_DIR = MODELS_DIR / RUN_NAME
OUTPUT_RUN_CONFIG_PATH = OUTPUT_DIR / "run_config.json"

LOG_DIR = PROJECT_ROOT / "csv" / "loggings" / "sft" / "heatmap_analysis"
LOG_PATH = LOG_DIR / f"train_{RUN_NAME}.log"
RUN_CONFIG_PATH = LOG_DIR / f"run_config_{RUN_NAME}.json"

WANDB_DIR = PROJECT_ROOT / "wandb"


def build_run_config(local_rank: int, is_main_process: bool) -> dict[str, Any]:
    return {
        "run_name": RUN_NAME,
        "mode": MODE,
        "experiment_tag": EXPERIMENT_TAG,
        "base_model_name": BASE_MODEL_NAME,
        "model_id": MODEL_ID,
        "wandb_project": WANDB_PROJECT,
        "paths": {
            "project_root": PROJECT_ROOT,
            "train_csv": TRAIN_CSV_PATH,
            "val_csv": VAL_CSV_PATH,
            "prompt_path": PROMPT_PATH,
            "models_dir": MODELS_DIR,
            "output_dir": OUTPUT_DIR,
            "log_dir": LOG_DIR,
            "log_path": LOG_PATH,
            "run_config_path": RUN_CONFIG_PATH,
            "wandb_dir": WANDB_DIR,
        },
        "data": {
            "train_sample_frac": TRAIN_SAMPLE_FRAC,
            "val_sample_frac": VAL_SAMPLE_FRAC,
            "sample_random_seed": SAMPLE_RANDOM_SEED,
        },
        "resume_from_checkpoint": RESUME_FROM_CHECKPOINT,
        "training": {
            "num_train_epochs": NUM_TRAIN_EPOCHS,
            "per_device_train_batch_size": PER_DEVICE_TRAIN_BATCH_SIZE,
            "per_device_eval_batch_size": PER_DEVICE_EVAL_BATCH_SIZE,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "learning_rate": LEARNING_RATE,
            "logging_steps": LOGGING_STEPS,
            "eval_steps": EVAL_STEPS,
            "save_strategy": SAVE_STRATEGY,
            "eval_strategy": EVAL_STRATEGY,
            "enable_early_stopping": ENABLE_EARLY_STOPPING,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "early_stopping_threshold": EARLY_STOPPING_THRESHOLD,
            "best_model_metric": BEST_MODEL_METRIC,
            "best_model_greater_is_better": BEST_MODEL_GREATER_IS_BETTER,
            "push_to_hub": PUSH_TO_HUB,
            "hub_private_repo": HUB_PRIVATE_REPO,
        },
        "lora": {
            "r": LORA_R,
            "alpha": LORA_ALPHA,
            "dropout": LORA_DROPOUT,
            "bias": LORA_BIAS,
            "target_modules": LORA_TARGET_MODULES,
            "task_type": LORA_TASK_TYPE,
            "modules_to_save": LORA_MODULES_TO_SAVE,
        },
        "distributed": {
            "local_rank": local_rank,
            "is_main_process": is_main_process,
        },
    }
