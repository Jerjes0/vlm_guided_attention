import json
import wandb
import torch
import numpy as np

from transformers import TrainerCallback


class WandBPredictionLogger(TrainerCallback):
    """
    SFT callback: logs prediction tables on eval.
    Works for both single-image and multi-image samples.
    """

    def __init__(self, dataset, processor, num_samples=2, max_new_tokens=256):
        self.dataset = dataset
        self.processor = processor
        self.num_samples = num_samples
        self.max_new_tokens = max_new_tokens

    def on_evaluate(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        if wandb.run is None:
            return

        model = kwargs["model"]
        model.eval()
        device = next(model.parameters()).device

        rows = []
        max_images_seen = 0

        for i in range(self.num_samples):
            example = self.dataset[i]
            images = example["images"]
            messages = example["messages"]
            max_images_seen = max(max_images_seen, len(images))

            prompt = self.processor.apply_chat_template(
                messages[:-1],  # system + user only
                add_generation_prompt=True,
                tokenize=False,
            )

            inputs = self.processor(
                text=[prompt],
                images=[images],
                return_tensors="pt",
                padding=True,
            ).to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                )

            generated = self.processor.decode(outputs[0], skip_special_tokens=True)
            prediction = generated[len(prompt):].strip() if generated.startswith(prompt) else generated.strip()
            target = messages[-1]["content"][0]["text"]

            image_cells = [wandb.Image(img, caption=f"image_{k}") for k, img in enumerate(images)]
            rows.append((image_cells, prediction, target))

        image_columns = [f"image_{k}" for k in range(max_images_seen)]
        table_columns = [*image_columns, "prediction", "ground_truth"]

        table_data = []
        for image_cells, prediction, target in rows:
            padded_images = image_cells + [None] * (max_images_seen - len(image_cells))
            table_data.append([*padded_images, prediction, target])

        table = wandb.Table(columns=table_columns, data=table_data)
        wandb.log({"predictions": table}, step=state.global_step)


class WandBGRPOPredictionLogger(TrainerCallback):
    """
    GRPO callback: periodically logs image + completion + judge parse + reward table.
    """

    def __init__(
        self,
        dataset,
        processor,
        reward_model,
        num_samples=4,
        max_new_tokens=128,
        log_every_n_steps=50,
    ):
        self.dataset = dataset
        self.processor = processor
        self.reward_model = reward_model
        self.num_samples = num_samples
        self.max_new_tokens = max_new_tokens
        self.log_every_n_steps = log_every_n_steps
        self._last_logged_step = -1

    def on_log(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        if state.global_step <= 0:
            return
        if state.global_step % self.log_every_n_steps != 0:
            return
        if state.global_step == self._last_logged_step:
            return

        model = kwargs["model"]
        model.eval()
        device = next(model.parameters()).device
        rows = []
        max_images_seen = 0

        with torch.no_grad():
            for i in range(min(self.num_samples, len(self.dataset))):
                example = self.dataset[i]
                prompt_messages = example["prompt"]
                images = example["images"]
                reference_text = example["reference_text"]
                max_images_seen = max(max_images_seen, len(images))

                prompt_text = self.processor.apply_chat_template(
                    prompt_messages,
                    add_generation_prompt=True,
                    tokenize=False,
                )

                inputs = self.processor(
                    text=[prompt_text],
                    images=[images],
                    return_tensors="pt",
                    padding=True,
                ).to(device)

                outputs = model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    top_p=0.95,
                    temperature=1.0,
                )

                generated = self.processor.decode(outputs[0], skip_special_tokens=True)
                prediction_text = generated.rsplit("model\n", 1)[-1].strip() if "model\n" in generated else generated.strip()

                parsed_prediction = self.reward_model._run_judge(text=prediction_text)
                parsed_ground_truth = self.reward_model._run_judge(text=reference_text)
                reward_value = self.reward_model._compute_reward(
                    parsed_prediction,
                    parsed_ground_truth,
                    generation_text=prediction_text,
                )

                image_cells = [wandb.Image(img, caption=f"image_{k}") for k, img in enumerate(images)]
                rows.append(
                    (
                        image_cells,
                        prediction_text,
                        reference_text,
                        json.dumps(parsed_prediction, ensure_ascii=True),
                        json.dumps(parsed_ground_truth, ensure_ascii=True),
                        float(reward_value),
                    )
                )

        image_columns = [f"image_{k}" for k in range(max_images_seen)]
        table_columns = [
            *image_columns,
            "prediction",
            "ground_truth",
            "judge_prediction",
            "judge_ground_truth",
            "reward",
        ]

        table_data = []
        for image_cells, prediction, ground_truth, judge_pred, judge_gt, reward in rows:
            padded_images = image_cells + [None] * (max_images_seen - len(image_cells))
            table_data.append([*padded_images, prediction, ground_truth, judge_pred, judge_gt, reward])

        table = wandb.Table(columns=table_columns, data=table_data)
        wandb.log({"grpo_predictions": table, "global_step": state.global_step})
        self._last_logged_step = state.global_step


class GRPOHeldoutRewardEvaluator(TrainerCallback):
    """
    Evaluates fixed held-out GRPO samples and logs reward/termination diagnostics.
    """

    def __init__(
        self,
        dataset,
        processor,
        reward_model,
        num_samples=100,
        max_new_tokens=64,
        log_every_n_steps=100,
    ):
        self.dataset = dataset
        self.processor = processor
        self.reward_model = reward_model
        self.num_samples = num_samples
        self.max_new_tokens = max_new_tokens
        self.log_every_n_steps = log_every_n_steps
        self._last_logged_step = -1

    def on_log(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        if state.global_step <= 0:
            return
        if state.global_step % self.log_every_n_steps != 0:
            return
        if state.global_step == self._last_logged_step:
            return

        model = kwargs["model"]
        model.eval()
        device = next(model.parameters()).device

        prompt_texts = []
        completion_texts = []
        reference_texts = []
        clipped = 0
        terminated = 0
        total = 0

        eos_token_id = self.processor.tokenizer.eos_token_id
        pad_token_id = self.processor.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = eos_token_id

        with torch.no_grad():
            for i in range(min(self.num_samples, len(self.dataset))):
                example = self.dataset[i]
                prompt_messages = example["prompt"]
                images = example["images"]
                reference_text = example["reference_text"]

                prompt_text = self.processor.apply_chat_template(
                    prompt_messages,
                    add_generation_prompt=True,
                    tokenize=False,
                )

                inputs = self.processor(
                    text=[prompt_text],
                    images=[images],
                    return_tensors="pt",
                    padding=True,
                ).to(device)

                outputs = model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    eos_token_id=eos_token_id,
                    pad_token_id=pad_token_id,
                )

                generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
                has_eos = bool((generated_ids == eos_token_id).any().item()) if eos_token_id is not None else False
                hit_max_tokens = generated_ids.shape[0] >= self.max_new_tokens
                if has_eos:
                    terminated += 1
                if hit_max_tokens and not has_eos:
                    clipped += 1

                generated = self.processor.decode(outputs[0], skip_special_tokens=True)
                prediction_text = generated.rsplit("model\n", 1)[-1].strip() if "model\n" in generated else generated.strip()

                prompt_texts.append(prompt_text)
                completion_texts.append(prediction_text)
                reference_texts.append(reference_text)
                total += 1

        rewards = self.reward_model.score(
            prompts=prompt_texts,
            generations=completion_texts,
            references=reference_texts,
        )

        reward_mean = float(np.mean(rewards)) if rewards else 0.0
        reward_std = float(np.std(rewards)) if rewards else 0.0
        clipped_ratio = float(clipped / total) if total > 0 else 0.0
        terminated_ratio = float(terminated / total) if total > 0 else 0.0

        wandb.log(
            {
                "eval/reward_mean": reward_mean,
                "eval/reward_std": reward_std,
                "eval/completions_clipped_ratio": clipped_ratio,
                "eval/completions_terminated_ratio": terminated_ratio,
                "eval/num_samples": total,
                "global_step": state.global_step,
            }
        )
        self._last_logged_step = state.global_step
