# Reward.py

import torch
from typing import List, Dict, Any
from unsloth import FastLanguageModel
from peft import PeftModel

# ----------------- CONFIG -----------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_TOKENS = 128

JUDGE_MODEL_NAME = (
    "/home/jerjes/repos/CXR_LLM_Benchmark/finetuned_models/new/"
    "medgemma-27B-text-it-unsloth-bnb-4bit_short"
)

PROMPT_FILE = "/home/jerjes/repos/CXR_LLM_Benchmark/csv/prompts/Pneumonia_short.txt"

PROMPT_TEMPLATE = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
"""


class RewardModel:
    def __init__(self):
        # -------- Load instruction once --------
        with open(PROMPT_FILE, "r") as f:
            self.instruction = f.read().strip()

        # -------- Load judge model once --------
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=JUDGE_MODEL_NAME,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )

        self.model = PeftModel.from_pretrained(
            self.model,
            JUDGE_MODEL_NAME,
            is_trainable=False,
        )
        self.model = self.model.merge_and_unload()

        FastLanguageModel.for_inference(self.model)
        self.model.eval()

    @torch.no_grad()
    def score(
        self,
        prompts: List[str],
        generations: List[str],
        references: List[str],
    ) -> List[float]:
        """
        Computes PPO rewards by parsing VLM output and ground truth
        with the SAME judge prompt structure.
        """

        rewards = []

        for prompt, generation, reference in zip(prompts, generations, references):

            # -------- Parse VLM output --------
            parsed_vlm = self._run_judge(text=generation)

            # -------- Parse ground truth --------
            parsed_gt = self._run_judge(text=reference)

            # -------- Compute reward --------
            reward = self._compute_reward(parsed_vlm, parsed_gt)
            rewards.append(float(reward))

        return rewards

    def _run_judge(self, text: str) -> Dict[str, Any]:
        """
        Runs the judge LLM on a single piece of text using the
        EXACT same prompt structure as the benchmark script.
        """

        judge_prompt = PROMPT_TEMPLATE.format(
            self.instruction,
            text,
            "",
        )

        inputs = self.tokenizer(
            [judge_prompt],
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(DEVICE)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=MAX_TOKENS,
            do_sample=False,
            use_cache=True,
            pad_token_id=self.tokenizer.eos_token_id,
            temperature=None,
            top_p=None,
        )

        decoded = self.tokenizer.batch_decode(
            outputs,
            skip_special_tokens=True,
        )[0]

        response = decoded.split("### Response:")[-1].strip()

        return response #self._parse_structured_response(decoded)

    def _parse_structured_response(self, response: str) -> Dict[str, Any]:
        """
        TODO:
        Extract structured attributes, e.g.
        likelihood, focality, location, change
        """
        return {}

    def _compute_reward(
        self,
        parsed_vlm: Dict[str, Any],
        parsed_gt: Dict[str, Any],
    ) -> float:
        """
        TODO:
        Compare parsed attributes and return scalar reward.
        """
        return 0.0
