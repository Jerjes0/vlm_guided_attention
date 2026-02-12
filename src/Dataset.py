import h5py
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from typing import Any


def _normalize_xray_uint8(image_raw: np.ndarray) -> np.ndarray:
    image = image_raw.astype("float32")
    image = (image - image.min()) / (image.max() - image.min() + 1e-6)
    return (image * 255.0).astype("uint8")


def _make_heatmap_overlay_image(img_uint8: np.ndarray, heatmap_raw: np.ndarray) -> Image.Image:
    """
    Build a single RGB image with transparent-zero heatmap overlay.

    - Heatmap==0 -> fully transparent.
    - Non-zero heatmap -> yellow/orange overlay with low alpha.
    """
    base_rgb = np.array(Image.fromarray(img_uint8).convert("RGB")).astype("float32")

    # Keep only positive relevance; exact zeros remain transparent.
    heat = np.clip(heatmap_raw.astype("float32"), a_min=0.0, a_max=None)
    if heat.max() > 0:
        heat_norm = heat / (heat.max() + 1e-6)
    else:
        heat_norm = np.zeros_like(heat, dtype="float32")

    # Resize heatmap intensity map to image resolution.
    heat_pil = Image.fromarray((heat_norm * 255.0).astype("uint8"), mode="L")
    heat_resized = np.array(heat_pil.resize((1024, 1024), Image.BILINEAR)).astype("float32") / 255.0

    # Color ramp from yellow -> orange.
    # low intensity: [255, 220, 0], high intensity: [255, 140, 0]
    red = np.full_like(heat_resized, 255.0)
    green = 220.0 - 80.0 * heat_resized
    blue = np.zeros_like(heat_resized)
    heat_color = np.stack([red, green, blue], axis=-1)

    max_alpha = 0.30
    alpha = heat_resized * max_alpha
    alpha[heat_resized <= 1e-6] = 0.0
    alpha = alpha[..., None]

    overlay = base_rgb * (1.0 - alpha) + heat_color * alpha
    overlay = np.clip(overlay, 0, 255).astype("uint8")
    return Image.fromarray(overlay).convert("RGB")


class ChestXrayDataset(Dataset):
    def __init__(self, df, instruction_prompt, mode='image'):
        self.df = df.reset_index(drop=True)
        self.instruction_prompt = instruction_prompt
        self.mode = mode
        # Cache open HDF5 files (important for speed)
        self._hdf5_files = {}
    
    def _get_hdf5(self, hdf5_number):
        if hdf5_number not in self._hdf5_files:
            path = (
                f"/home/jerjes/mnt/maxwell_projects/"
                f"LLM_dataset_Chest_Xrays/data_{hdf5_number}_with_ensemble.hdf5"
            )
            self._hdf5_files[hdf5_number] = h5py.File(path, "r")
        return self._hdf5_files[hdf5_number]
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # ---------
        # Load image from HDF5
        # ---------
        f = self._get_hdf5(row.hdf5)
        key = f"{row.pid}__{row.study_id}__0"
        grp = f[key]
        
        img = _normalize_xray_uint8(grp["image_raw_1024"][()])
        
        # Build single overlay image when heatmap mode is enabled
        if self.mode == 'image_and_heatmap':
            heatmap_img = _make_heatmap_overlay_image(img, grp["Heatmap_256"][()])
        
        img = Image.fromarray(img).convert("RGB")  # Convert to PIL (MedGemma expects PIL)
        
        # ---------
        # Build messages
        # ---------
        if self.mode == 'image_and_heatmap':
            messages = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": self.instruction_prompt,
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {
                            "type": "text",
                            "text": (
                                "This chest X-ray image includes an overlaid attention heatmap. "
                                "Heatmap regions are transparent where relevance is zero and appear "
                                "in yellow/orange where relevance is higher. "
                                f"{row.exam_clinical}"
                            ),
                        },
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": row.findings_impression,
                        }
                    ],
                },
            ]
        else:
            messages = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": self.instruction_prompt,
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {
                            "type": "text",
                            "text": row.exam_clinical,
                        },
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": row.findings_impression,
                        }
                    ],
                },
            ]
        
        if self.mode == 'image_and_heatmap':
            return {
                "images": [heatmap_img],
                "messages": messages,
            }
        else:
            return {
                "images": [img],
                "messages": messages,
            }
    
    def close(self):
        for f in self._hdf5_files.values():
            f.close()


class ChestXrayPPODataset(Dataset):
    """
    PPO-specific dataset.

    - Prompt contains ONLY system + user messages
    - Ground truth report is returned separately as reference_text
    - Supports both image and image_and_heatmap modes
    """

    def __init__(self, df, instruction_prompt, mode="image"):
        self.df = df.reset_index(drop=True)
        self.instruction_prompt = instruction_prompt
        self.mode = mode
        self._hdf5_files = {}

    # ---------------------------------------------------------
    # HDF5 handling (same as SFT)
    # ---------------------------------------------------------

    def _get_hdf5(self, hdf5_number):
        if hdf5_number not in self._hdf5_files:
            path = (
                f"/home/jerjes/mnt/maxwell_projects/"
                f"LLM_dataset_Chest_Xrays/data_{hdf5_number}_with_ensemble.hdf5"
            )
            self._hdf5_files[hdf5_number] = h5py.File(path, "r")
        return self._hdf5_files[hdf5_number]

    def __len__(self):
        return len(self.df)

    # ---------------------------------------------------------
    # Main item loader
    # ---------------------------------------------------------

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # -------------------------
        # Load image
        # -------------------------

        f = self._get_hdf5(row.hdf5)
        key = f"{row.pid}__{row.study_id}__0"
        grp = f[key]

        img = _normalize_xray_uint8(grp["image_raw_1024"][()])
        img = Image.fromarray(img).convert("RGB")

        images = [img]

        # -------------------------
        # Optional heatmap
        # -------------------------

        if self.mode == "image_and_heatmap":
            heatmap_img = _make_heatmap_overlay_image(np.array(img.convert("L")), grp["Heatmap_256"][()])
            images = [heatmap_img]

        # -------------------------
        # Build PPO prompt (NO assistant)
        # -------------------------

        if self.mode == "image_and_heatmap":
            user_text = (
                "This chest X-ray image includes an overlaid attention heatmap. "
                "Heatmap regions are transparent where relevance is zero and appear "
                "in yellow/orange where relevance is higher. "
                f"{row.exam_clinical}"
            )
            user_content = [
                {"type": "image"},
                {"type": "text", "text": user_text},
            ]
        else:
            user_content = [
                {"type": "image"},
                {"type": "text", "text": row.exam_clinical},
            ]

        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": self.instruction_prompt}
                ],
            },
            {
                "role": "user",
                "content": user_content,
            },
        ]

        # -------------------------
        # Ground truth (for reward)
        # -------------------------

        reference_text = row.findings_impression

        return {
            "images": images,
            "messages": messages,
            "reference_text": reference_text,
        }

    def close(self):
        for f in self._hdf5_files.values():
            f.close()


class ChestXrayGRPODataset(Dataset):
    """
    Adapter from ChestXrayPPODataset to TRL GRPO expected schema.

    Each row contains:
    - prompt: conversational messages (system + user only)
    - images: list[PIL.Image]
    - reference_text: used by custom reward function
    """

    def __init__(self, base_dataset: ChestXrayPPODataset, single_image_for_grpo: bool = True):
        self.base_dataset = base_dataset
        self.single_image_for_grpo = single_image_for_grpo

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ex = self.base_dataset[idx]
        prompt = ex["messages"]
        images = ex["images"]

        # Some TRL+VLM combinations can mismatch multi-image placeholders vs extracted
        # features in GRPO loss computation. For stability, collapse to a single image
        # (prefer the heatmap overlay image when available) and keep one image placeholder.
        if self.single_image_for_grpo and len(images) > 1:
            images = [images[-1]]
            prompt = self._force_single_image_placeholder(prompt)

        return {
            "prompt": prompt,
            "images": images,
            "reference_text": ex["reference_text"],
        }

    def _force_single_image_placeholder(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        normalized = []
        user_fixed = False

        for msg in messages:
            if (
                not user_fixed
                and isinstance(msg, dict)
                and msg.get("role") == "user"
                and isinstance(msg.get("content"), list)
            ):
                text_blocks = [b for b in msg["content"] if isinstance(b, dict) and b.get("type") == "text"]
                # Keep exactly one image placeholder + all text blocks.
                fixed_content = [{"type": "image"}, *text_blocks]
                normalized.append({**msg, "content": fixed_content})
                user_fixed = True
            else:
                normalized.append(msg)

        return normalized

    def close(self) -> None:
        if hasattr(self.base_dataset, "close"):
            self.base_dataset.close()
