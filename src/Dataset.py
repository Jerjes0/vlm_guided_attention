import h5py
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

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
        
        img = grp["image_raw_1024"][()].astype("float32")
        img = (img - img.min()) / (img.max() - img.min() + 1e-6)
        img = (img * 255.0).astype("uint8")
        
        # Get heatmap
        if self.mode == 'image_and_heatmap':
            heatmap = grp["Heatmap_256"][()]
            
            # Normalize heatmap to 0-1 range
            heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-6)
            
            # Apply colormap (red for high values)
            cmap = plt.get_cmap('jet')
            heatmap_colored = cmap(heatmap_norm)  # Returns RGBA
            heatmap_rgb = (heatmap_colored[:, :, :3] * 255).astype('uint8')
            
            # Resize heatmap to match original image (1024x1024)
            heatmap_pil = Image.fromarray(heatmap_rgb).convert("RGB")
            heatmap_resized = heatmap_pil.resize((1024, 1024), Image.LANCZOS)
            heatmap_array = np.array(heatmap_resized)
            
            # Get original image as array
            img_rgb = Image.fromarray(img).convert("RGB")
            img_array = np.array(img_rgb)
            
            # Create overlay with higher alpha for better visibility
            alpha = 0.6  # Increased for better heatmap visibility
            overlay = (alpha * heatmap_array + (1 - alpha) * img_array).astype('uint8')
            
            # Convert to PIL Image
            heatmap_img = Image.fromarray(overlay).convert("RGB")
        
        img = Image.fromarray(img).convert("RGB")  # Convert to PIL (MedGemma expects PIL)
        
        # ---------
        # Build messages
        # ---------
        if self.mode == 'image_and_heatmap':
            # Just pass both images but treat them as a single multi-image input
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
                        {"type": "image"},  # First image
                        {"type": "image"},  # Second image
                        {
                            "type": "text",
                            "text": f"The first image shows the original chest X-ray. The second image shows the same X-ray with a heatmap overlay highlighting areas of interest. {row.exam_clinical}",
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
                "images": [img, heatmap_img],
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