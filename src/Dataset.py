import h5py
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class ChestXrayDataset(Dataset):
    def __init__(self, df, instruction_prompt):
        self.df = df.reset_index(drop=True)
        self.instruction_prompt = instruction_prompt

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
        img = grp["image_raw_1024"][()].astype("float32")
        img = (img - img.min()) / (img.max() - img.min() + 1e-6)
        img = (img * 255.0).astype("uint8")
        img = Image.fromarray(img).convert("RGB") # Convert to PIL (MedGemma expects PIL)

        # ---------
        # Build messages
        # ---------
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

        if 
        return {
            "image": img,
            "messages": messages,
        }

    def close(self):
        for f in self._hdf5_files.values():
            f.close()
