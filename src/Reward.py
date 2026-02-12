# Reward.py

import json
import math
import re
from typing import Dict, Any, List

import torch
# from unsloth import FastLanguageModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


# ----------------- CONFIG -----------------

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
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
    def __init__(self, device):
        self.device = device
        self.device_str = str(device) if isinstance(device, torch.device) else str(device)
        # -------- Load instruction once --------
        with open(PROMPT_FILE, "r") as f:
            self.instruction = f.read().strip()
            self.base_model_name = "unsloth/medgemma-27B-text-it-unsloth-bnb-4bit"
            self.judge_model_name = (
                "/home/jerjes/repos/CXR_LLM_Benchmark/finetuned_models/new/"
                "medgemma-27B-text-it-unsloth-bnb-4bit_short"
            )

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            quantization_config=quantization_config,
            device_map={"": self.device_str},
        )

        self.model = PeftModel.from_pretrained(
            base_model,
            self.judge_model_name,
            is_trainable=False,
        )

        self.model = self.model.merge_and_unload()
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.judge_model_name)


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
        ).to(self.device)

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

        return self._parse_structured_response(response)

    def _parse_structured_response(self, response: str) -> Dict[str, Any]:
        """
        Parse LLM output expected to contain a JSON object with keys like:
        Likelihood, Focality, Location(s), Change(s)
        """

        # 1) Remove code fences if present
        cleaned = response.strip()
        cleaned = re.sub(r"^```json", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"^```", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()

        # 2) Extract JSON substring (in case of extra text)
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match is None:
            return {}

        json_str = match.group(0)

        # 3) Parse JSON
        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError:
            return {}

        # 4) Normalize keys (optional but strongly recommended)
        normalized = {}

        key_map = {
            "Likelihood": "likelihood",
            "Focality": "focality",
            "Location(s)": "location",
            "Locations": "location",
            "Change(s)": "change",
            "Changes": "change",
        }

        for k, v in parsed.items():
            key = key_map.get(k, k.lower())
            normalized[key] = v

        return normalized


    def _compute_reward(
        self,
        parsed_vlm: Dict[str, Any],
        parsed_gt: Dict[str, Any],
    ) -> float:
        
        if not parsed_vlm or not parsed_gt:
            return -1.0

        # =========================================================
        # Reward coefficients (TUNABLE HYPERPARAMETERS)
        # =========================================================

        # Likelihood (most important)
        ALPHA_FALSE_POSITIVE = 0.8     # GT = none, prediction > none
        ALPHA_LIKELIHOOD_DIST = 0.25   # ordinal distance penalty
        BETA_LIKELIHOOD_MATCH = 0.6    # exact likelihood match reward

        # Internal consistency
        ALPHA_INCONSISTENT_FOCALITY = 0.4
        ALPHA_INCONSISTENT_LOCATION = 0.4

        # Focality
        BETA_FOCALITY_MATCH = 0.2
        ALPHA_FOCALITY_MISMATCH = 0.2

        # Location
        BETA_LOCATION_EXACT_MATCH = 0.3
        ALPHA_LOCATION_MISSING = 0.15
        ALPHA_LOCATION_INVENTED = 0.35

        # =========================================================
        # Likelihood (ORDINAL)
        # =========================================================

        reward = 0.0

        LIKELIHOOD_MAP = {
            "none": 0,
            "low": 1,
            "medium": 2,
            "high": 3,
        }

        l_gt = LIKELIHOOD_MAP.get(
            str(parsed_gt.get("likelihood", "none")).lower(), 0
        )
        l_vlm = LIKELIHOOD_MAP.get(
            str(parsed_vlm.get("likelihood", "none")).lower(), 0
        )

        # False positive: very strong penalty
        if l_gt == 0 and l_vlm > 0:
            reward -= ALPHA_FALSE_POSITIVE * l_vlm

        # Ground truth present: ordinal disagreement
        elif l_gt > 0:
            dist = abs(l_vlm - l_gt)
            reward -= ALPHA_LIKELIHOOD_DIST * dist

            # Exact match reward
            if dist == 0:
                reward += BETA_LIKELIHOOD_MATCH

        # =========================================================
        # Internal consistency
        # =========================================================

        if l_vlm == 0:
            if parsed_vlm.get("focality") not in [None, "", "NA", "unknown"]:
                reward -= ALPHA_INCONSISTENT_FOCALITY
            if parsed_vlm.get("location") not in [None, "", "NA", "unknown"]:
                reward -= ALPHA_INCONSISTENT_LOCATION

        # =========================================================
        # Focality
        # =========================================================

        focality_gt = str(parsed_gt.get("focality", "")).lower()
        focality_vlm = str(parsed_vlm.get("focality", "")).lower()

        if focality_gt and focality_vlm:
            if focality_gt == focality_vlm:
                reward += BETA_FOCALITY_MATCH
            else:
                reward -= ALPHA_FOCALITY_MISMATCH

        # =========================================================
        # Location
        # =========================================================

        def _split_locations(x):
            if x in [None, "", "NA", "unknown"]:
                return set()
            return {loc.strip().lower() for loc in str(x).split(",")}

        gt_locs = _split_locations(parsed_gt.get("location"))
        vlm_locs = _split_locations(parsed_vlm.get("location"))

        missing = gt_locs - vlm_locs
        invented = vlm_locs - gt_locs

        # Exact location match reward
        if gt_locs and gt_locs == vlm_locs:
            reward += BETA_LOCATION_EXACT_MATCH
        else:
            reward -= ALPHA_LOCATION_MISSING * len(missing)
            reward -= ALPHA_LOCATION_INVENTED * len(invented)

        # =========================================================
        # Change (ignored for now)
        # =========================================================

        # =========================================================
        # PPO-friendly clipping
        # =========================================================

        
        reward = math.tanh(reward / 0.7)

        # reward = max(min(reward, 1.0), -1.0)

        return reward
