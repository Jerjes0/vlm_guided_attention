import json
import logging
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from Reward import RewardModel


# ------------------------------------------------------------
# Run configuration (edit only here)
# ------------------------------------------------------------
CSV_PATH = Path(
    "/home/jerjes/repos/visual_guided_attention/csv/outputs/heatmap_analysis/predictions_ft_lora_image_and_heatmap_2026-02-28_12-40-39.csv"
)
OUTPUT_CSV: Path | None = None
PROCESS_GROUND_TRUTH = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_EVERY = 10


def to_json_cell(parsed: dict) -> str:
    return json.dumps(parsed, ensure_ascii=False, sort_keys=True)


def preprocess_prediction_text(text: str) -> str:
    # Keep only the assistant output after the last "model" marker.
    return text.split("model")[-1].strip()


def main() -> None:
    if SAVE_EVERY < 1:
        raise ValueError("SAVE_EVERY must be >= 1")

    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    output_csv = OUTPUT_CSV or CSV_PATH.with_name(f"{CSV_PATH.stem}_evaluated.csv")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    log = logging.getLogger("evaluate")

    log.info("Loading CSV: %s", CSV_PATH)
    df = pd.read_csv(CSV_PATH)
    if "prediction" not in df.columns:
        raise ValueError('Input CSV must contain a "prediction" column.')
    if PROCESS_GROUND_TRUTH and "ground_truth" not in df.columns:
        raise ValueError('PROCESS_GROUND_TRUTH=True, but "ground_truth" column is missing.')

    log.info("Loading RewardModel judge on device: %s", DEVICE)
    reward_model = RewardModel(device=DEVICE)

    prediction_cols = [
        "prediction_llm_parsed",
        "prediction_llm_likelihood",
        "prediction_llm_focality",
        "prediction_llm_location",
        "prediction_llm_change",
    ]
    for col in prediction_cols:
        if col not in df.columns:
            df[col] = None

    if PROCESS_GROUND_TRUTH:
        gt_cols = [
            "ground_truth_llm_parsed",
            "ground_truth_llm_likelihood",
            "ground_truth_llm_focality",
            "ground_truth_llm_location",
            "ground_truth_llm_change",
        ]
        for col in gt_cols:
            if col not in df.columns:
                df[col] = None

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    log.info("Created output CSV (initial state): %s", output_csv)

    total_rows = len(df)
    log.info("Running judge incrementally on %d rows", total_rows)

    for idx in tqdm(range(total_rows), desc="Evaluating"):
        raw_pred_text = str(df.at[idx, "prediction"]) if pd.notna(df.at[idx, "prediction"]) else ""
        pred_text = preprocess_prediction_text(raw_pred_text)
        pred_parsed = reward_model.parse_text(pred_text)

        df.at[idx, "prediction_llm_parsed"] = to_json_cell(pred_parsed)
        df.at[idx, "prediction_llm_likelihood"] = pred_parsed.get("likelihood")
        df.at[idx, "prediction_llm_focality"] = pred_parsed.get("focality")
        df.at[idx, "prediction_llm_location"] = pred_parsed.get("location")
        df.at[idx, "prediction_llm_change"] = pred_parsed.get("change")

        if PROCESS_GROUND_TRUTH:
            gt_text = str(df.at[idx, "ground_truth"]) if pd.notna(df.at[idx, "ground_truth"]) else ""
            gt_parsed = reward_model.parse_text(gt_text)
            df.at[idx, "ground_truth_llm_parsed"] = to_json_cell(gt_parsed)
            df.at[idx, "ground_truth_llm_likelihood"] = gt_parsed.get("likelihood")
            df.at[idx, "ground_truth_llm_focality"] = gt_parsed.get("focality")
            df.at[idx, "ground_truth_llm_location"] = gt_parsed.get("location")
            df.at[idx, "ground_truth_llm_change"] = gt_parsed.get("change")

        if (idx + 1) % SAVE_EVERY == 0 or (idx + 1) == total_rows:
            df.to_csv(output_csv, index=False)

    log.info("Saved evaluated CSV: %s", output_csv)


if __name__ == "__main__":
    main()
