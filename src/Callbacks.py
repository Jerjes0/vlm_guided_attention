
import wandb

import torch



from transformers import TrainerCallback


class WandBPredictionLogger(TrainerCallback):
    def __init__(self, dataset, processor, num_samples=2, max_new_tokens=256):
        self.dataset = dataset
        self.processor = processor
        self.num_samples = num_samples
        self.max_new_tokens = max_new_tokens
        self.mode = dataset.mode

    def on_evaluate(self, args, state, control, **kwargs):
        model = kwargs["model"]
        model.eval()

        rows = []

        for i in range(self.num_samples):
            example = self.dataset[i]

            images = example["images"]
            messages = example["messages"]

            prompt = self.processor.apply_chat_template(
                messages[:-1],  # system + user only
                add_generation_prompt=True,
                tokenize=False,
            )

            inputs = self.processor(
                text=[prompt],
                images=[images],
                return_tensors="pt",
                padding=True
            ).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                )

            generated = self.processor.decode(
                outputs[0],
                skip_special_tokens=True
            )

            if generated.startswith(prompt):
                prediction = generated[len(prompt):].strip()
            else:
                prediction = generated.strip()

            target = messages[-1]["content"][0]["text"]

            if self.mode == 'image_and_heatmap':
                rows.append([
                    wandb.Image(images[0], caption="Original X-ray"),
                    wandb.Image(images[1], caption="Heatmap Overlay"),
                    prediction,
                    target,
                ])
            else:
                rows.append([
                    wandb.Image(images[0], caption="Original X-ray"),
                    prediction,
                    target,
                ])

        # Create table with appropriate columns based on mode
        if self.mode == 'image_and_heatmap':
            table = wandb.Table(
                columns=["original_image", "heatmap_image", "prediction", "ground_truth"],
                data=rows,
            )
        else:
            table = wandb.Table(
                columns=["image", "prediction", "ground_truth"],
                data=rows,
            )

        wandb.log(
            {
                "predictions": table,
                "global_step": state.global_step,
            }
        )
