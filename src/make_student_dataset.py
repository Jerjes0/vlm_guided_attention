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
from Reward import RewardModel

# ------------------------------------------------------------
# Run configuration (edit only here)
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODE = "image_and_heatmap"  # "image" or "image_and_heatmap"
STATE = "ft"
FT_STATE = "lora"

# Input dataset used to build student distillation data.
DATA_CSV_PATH = PROJECT_ROOT / "csv" / "train.csv"
NUM_SAMPLES: int | None = None  # Set to int for subset, None for full CSV

# Can be a hub model ID or a local run path under models/.
MODEL_CHECKPOINT: str | Path = (
    PROJECT_ROOT
    / "models"
    / "medgemma-1.5-4b-it-single_image_overlay-image_and_heatmap-lora-2026-02-12_22-00-01"
)

PROMPT_PATH = PROJECT_ROOT / "csv" / "prompts" / f"{MODE}.txt"

# Student-target sampling params
NUM_GENERATIONS = 8
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.9
TOP_P = 0.95

# VLM generation device (keep separate from judge for stability)
VLM_DEVICE_STR = "cuda:0"

# Judge device (RewardModel LLM judge)
JUDGE_DEVICE_STR = "cuda:1"

# Resume from existing output CSV (if provided).
RESUME_FROM_CSV: Path | None = None

RUN_TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
RUN_NAME = f"make-student-{STATE}-{FT_STATE}-{MODE}-{RUN_TIMESTAMP}"

LOG_DIR = PROJECT_ROOT / "csv" / "loggings" / "student_dataset"
LOG_PATH = LOG_DIR / f"{RUN_NAME}.log"
RUN_CONFIG_PATH = LOG_DIR / f"run_config_{RUN_NAME}.json"

OUTPUT_DIR = PROJECT_ROOT / "csv" / "outputs"
OUTPUT_CSV = OUTPUT_DIR / f"student_dataset_{STATE}_{FT_STATE}_{MODE}_{RUN_TIMESTAMP}.csv"
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
    checkpoint_path = Path(model_checkpoint) if isinstance(model_checkpoint, Path) else Path(str(model_checkpoint))

    # If a local path is given and points to a GRPO run directory, auto-select
    # latest checkpoint-* folder when adapter_config.json is not at root.
    if checkpoint_path.exists():
        checkpoint_path = checkpoint_path.resolve()
        if checkpoint_path.is_dir() and not (checkpoint_path / "adapter_config.json").exists():
            checkpoint_dirs = []
            for child in checkpoint_path.iterdir():
                if child.is_dir() and child.name.startswith("checkpoint-"):
                    try:
                        step = int(child.name.split("-", 1)[1])
                        checkpoint_dirs.append((step, child))
                    except (IndexError, ValueError):
                        continue
            if checkpoint_dirs:
                latest_checkpoint = max(checkpoint_dirs, key=lambda item: item[0])[1]
                return str(latest_checkpoint)
        return str(checkpoint_path)

    return str(model_checkpoint)


def save_run_config(config: dict[str, Any]) -> None:
    with open(RUN_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    with open(OUTPUT_RUN_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def sample_key_from_row(row: pd.Series | dict[str, Any]) -> str:
    row_dict = row if isinstance(row, dict) else row.to_dict()
    return "|".join(
        [
            str(row_dict.get("accession", "")),
            str(row_dict.get("pid", "")),
            str(row_dict.get("study_id", "")),
            str(row_dict.get("hdf5", "")),
        ]
    )


def atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp_path, index=False)
    tmp_path.replace(path)


LOG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()],
)
log = logging.getLogger("medgemma-make-student-dataset")

resolved_model_checkpoint = resolve_model_checkpoint(MODEL_CHECKPOINT)
run_config = {
    "run_name": RUN_NAME,
    "mode": MODE,
    "state": STATE,
    "ft_state": FT_STATE,
    "model_checkpoint": MODEL_CHECKPOINT,
    "resolved_model_checkpoint": resolved_model_checkpoint,
    "generation": {
        "num_generations": NUM_GENERATIONS,
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "num_samples": NUM_SAMPLES,
        "do_sample": True,
    },
    "paths": {
        "project_root": PROJECT_ROOT,
        "data_csv_path": DATA_CSV_PATH,
        "prompt_path": PROMPT_PATH,
        "log_dir": LOG_DIR,
        "log_path": LOG_PATH,
        "run_config_path": RUN_CONFIG_PATH,
        "output_dir": OUTPUT_DIR,
        "output_csv": OUTPUT_CSV,
        "resume_from_csv": RESUME_FROM_CSV,
    },
    "judge": {
        "judge_device": JUDGE_DEVICE_STR,
    },
    "vlm": {
        "vlm_device": VLM_DEVICE_STR,
    },
}
run_config_jsonable = _jsonable(run_config)
save_run_config(run_config_jsonable)

log.info("Starting student dataset generation script")
log.info("Run config:\n%s", json.dumps(run_config_jsonable, indent=2))

# ------------------------------------------------------------
# Load dataframe
# ------------------------------------------------------------
log.info("Loading source CSV: %s", DATA_CSV_PATH)
df = pd.read_csv(DATA_CSV_PATH)
if NUM_SAMPLES is not None:
    df = df.head(NUM_SAMPLES)
    log.info("Using first %d samples", NUM_SAMPLES)

run_config_jsonable["data"] = {"samples": len(df)}
save_run_config(run_config_jsonable)
log.info("Input samples: %d", len(df))

# ------------------------------------------------------------
# Load prompt + dataset
# ------------------------------------------------------------
log.info("Loading instruction prompt from %s", PROMPT_PATH)
with open(PROMPT_PATH, "r", encoding="utf-8") as f:
    instruction_prompt = f.read().strip()

log.info("Creating dataset in mode=%s", MODE)
dataset = ChestXrayDataset(df, instruction_prompt, mode=MODE)

# ------------------------------------------------------------
# Load VLM + processor
# ------------------------------------------------------------
log.info("Loading generation model from: %s", resolved_model_checkpoint)
if not torch.cuda.is_available():
    raise ValueError("CUDA is required for this script.")
if torch.cuda.get_device_capability()[0] < 8:
    raise ValueError("GPU does not support bfloat16, please use a GPU that supports bfloat16.")

pt_pipe = pipeline(
    "image-text-to-text",
    model=resolved_model_checkpoint,
    torch_dtype=torch.bfloat16,
    device_map={"": VLM_DEVICE_STR},
)
processor = AutoProcessor.from_pretrained(resolved_model_checkpoint)
pt_pipe.model.generation_config.pad_token_id = processor.tokenizer.eos_token_id

log.info("Loading judge model on %s", JUDGE_DEVICE_STR)
judge_device = torch.device(JUDGE_DEVICE_STR)
reward_model = RewardModel(device=judge_device)

# ------------------------------------------------------------
# Resume handling
# ------------------------------------------------------------
output_csv_path = RESUME_FROM_CSV if RESUME_FROM_CSV is not None else OUTPUT_CSV

rows: list[dict[str, Any]] = []
processed_keys: set[str] = set()
if output_csv_path.exists():
    log.info("Resuming from existing CSV: %s", output_csv_path)
    existing_df = pd.read_csv(output_csv_path)
    rows = existing_df.to_dict(orient="records")
    for existing_row in rows:
        processed_keys.add(sample_key_from_row(existing_row))
    log.info("Loaded %d existing rows", len(rows))

# Parse each unique GT once per run.
gt_parse_cache: dict[str, dict[str, Any]] = {}

# ------------------------------------------------------------
# Main generation loop
# ------------------------------------------------------------
log.info("Generating student dataset rows")
for idx in tqdm(range(len(dataset)), desc="Building student data"):
    source_row = df.iloc[idx]
    sample_key = sample_key_from_row(source_row)

    if sample_key in processed_keys:
        continue

    example = dataset[idx]

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
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            num_return_sequences=NUM_GENERATIONS,
        )

    generations: list[str] = []
    for seq in outputs:
        generated_text = processor.decode(seq, skip_special_tokens=True)
        if generated_text.startswith(prompt_text):
            prediction = generated_text[len(prompt_text):].strip()
        else:
            prediction = generated_text.strip()
        generations.append(prediction)

    ground_truth = example["messages"][-1]["content"][0]["text"]

    generation_judges = reward_model.parse_texts(generations)

    if ground_truth not in gt_parse_cache:
        gt_parse_cache[ground_truth] = reward_model.parse_text(ground_truth)
    gt_judge = gt_parse_cache[ground_truth]

    out_row = source_row.to_dict()
    out_row["sample_key"] = sample_key
    out_row["ground_truth"] = ground_truth
    out_row["ground_truth_judge_json"] = json.dumps(gt_judge, ensure_ascii=False)
    out_row["generations_json"] = json.dumps(generations, ensure_ascii=False)
    out_row["generation_judges_json"] = json.dumps(generation_judges, ensure_ascii=False)

    for i in range(NUM_GENERATIONS):
        out_row[f"generation_{i + 1}"] = generations[i] if i < len(generations) else ""
        judge_i = generation_judges[i] if i < len(generation_judges) else {}
        out_row[f"generation_{i + 1}_judge_json"] = json.dumps(judge_i, ensure_ascii=False)

    rows.append(out_row)
    processed_keys.add(sample_key)

    # Save every iteration so failures can resume without losing progress.
    atomic_write_csv(pd.DataFrame(rows), output_csv_path)

    if (len(rows) % 10) == 0:
        log.info("Saved %d rows to %s", len(rows), output_csv_path)

log.info("Finished generation. Total rows written: %d", len(rows))
log.info("Final output: %s", output_csv_path)

# ------------------------------------------------------------
# Cleanup
# ------------------------------------------------------------
log.info("Cleaning up resources")
dataset.close()
del pt_pipe
del reward_model
torch.cuda.empty_cache()
log.info("Done")
