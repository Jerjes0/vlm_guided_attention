# visual_guided_attention

Vision-language training pipeline for chest X-ray report generation with attention-overlay guidance, with a final layer of clinical reinforcement learning via large language model feedback.

## Overview
The project has two training stages:

1. SFT (supervised fine-tuning) of `google/medgemma-1.5-4b-it` with LoRA.
2. GRPO fine-tuning on top of the SFT adapter using a custom reward model.

Current experiment mode is `image_and_heatmap` with a **single overlay image**:
- The heatmap is overlaid on the X-ray.
- Heatmap value `0` is transparent.
- Non-zero heatmap appears in yellow/orange with low alpha.

## Architecture

### Data and preprocessing
- `src/Dataset.py`
  - `ChestXrayDataset`: SFT dataset (`messages`, `images`)
  - `ChestXrayPPODataset`: GRPO source dataset (`messages`, `images`, `reference_text`)
  - `ChestXrayGRPODataset`: TRL GRPO adapter dataset (`prompt`, `images`, `reference_text`)
  - Heatmap overlay creation and image normalization utilities

### SFT
- `src/finetune.py`
  - Loads train/val CSVs and prompt from `csv/prompts/<mode>.txt`
  - Builds SFT trainer (`trl.SFTTrainer`) with LoRA
  - Logs to W&B
  - Supports resume from checkpoint (`--resume-from-checkpoint`)

### RL (GRPO)
- `src/grpo_finetune.py`
  - Loads SFT adapter checkpoint
  - Uses `trl.GRPOTrainer`
  - Reward function delegates to `src/Reward.py`
  - Logs rollout samples/reward diagnostics to W&B

### Reward model
- `src/Reward.py`
  - Loads judge model
  - Parses structured outputs and computes scalar reward

### Callbacks and logging
- `src/Callbacks.py`
  - `WandBPredictionLogger` for SFT eval tables
  - `WandBGRPOPredictionLogger` for GRPO prediction/reward tables

### Inference
- `src/predict.py`
  - Batch inference on test split and CSV export

## Expected inputs
- `csv/train.csv`, `csv/val.csv`, `csv/test.csv`
- Prompt files in `csv/prompts/` (currently using `image_and_heatmap.txt`)
- HDF5 imaging files referenced by CSV rows

## Environment
Recommended env in this repo:
- `new_ppo_env`

Activate:

```bash
cd /home/jerjes/repos/visual_guided_attention/src
source ../new_ppo_env/bin/activate
```

Also make sure:
- `wandb login`
- `huggingface-cli login`

## Run order

### 1) SFT from scratch
Run from `src/`:

```bash
torchrun --nproc_per_node=2 finetune.py \
  --output-dir ../models/medgemma-1.5-4b-it-single_image_overlay-image_and_heatmap-lora \
  --eval-steps 25 \
  --logging-steps 25
```

### 2) (Optional) Resume SFT if interrupted
Resume latest checkpoint in the same output dir:

```bash
torchrun --nproc_per_node=2 finetune.py \
  --output-dir ../models/medgemma-1.5-4b-it-single_image_overlay-image_and_heatmap-lora \
  --resume-from-checkpoint auto \
  --eval-steps 25 \
  --logging-steps 25
```

Or resume a specific checkpoint:

```bash
torchrun --nproc_per_node=2 finetune.py \
  --output-dir ../models/medgemma-1.5-4b-it-single_image_overlay-image_and_heatmap-lora \
  --resume-from-checkpoint ../models/medgemma-1.5-4b-it-single_image_overlay-image_and_heatmap-lora/checkpoint-XXX
```

### 3) Run inference on test set (optional)

Edit `model_checkpoint` in `src/predict.py` to your trained adapter, then run:

```bash
python predict.py
```

### 4) Run GRPO on top of SFT adapter

1. Update `sft_checkpoint` in `src/grpo_finetune.py` to your new SFT output.
2. Run:

```bash
python grpo_finetune.py
```

## Outputs
- SFT artifacts: `--output-dir` path
- GRPO artifacts: directory created by `src/grpo_finetune.py`
- Logs: `csv/loggings_single_image_overlay/`
- W&B:
  - SFT project: `medgemma-chest-xray-single-image-overlay`
  - GRPO project: `medgemma-chest-xray-single-image-overlay-rl`
