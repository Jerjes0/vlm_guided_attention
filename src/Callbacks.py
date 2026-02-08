
import wandb

import torch



from transformers import TrainerCallback


class WandBPredictionLogger(TrainerCallback):
    def __init__(self, dataset, processor, num_samples=2, max_new_tokens=256):
        self.dataset = dataset
        self.processor = processor
        self.num_samples = num_samples
        self.max_new_tokens = max_new_tokens

    def on_evaluate(self, args, state, control, **kwargs):
        model = kwargs["model"]
        model.eval()

        rows = []

        for i in range(self.num_samples):
            example = self.dataset[i]

            image = example["image"]
            messages = example["messages"]

            prompt = self.processor.apply_chat_template(
                messages[:-1],  # system + user only
                add_generation_prompt=True,
                tokenize=False,
            )

            inputs = self.processor(
                text=[prompt],
                images=[[image]],
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

            rows.append([
                wandb.Image(image, caption="Input X-ray"),
                prediction,
                target,
            ])

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
