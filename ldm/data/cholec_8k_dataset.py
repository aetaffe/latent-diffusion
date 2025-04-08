from pathlib import Path

import PIL
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import albumentations

class Cholec8kDataset(Dataset):
    def __init__(self, root_dir, size):
        self.img_files = [str(file) for file in list(Path(root_dir).rglob("*[!mask].png"))]
        self.size = size

    def __len__(self):
        return len(self.img_files)

    def _get_mask(self, idx):
        filename = self.img_files[idx].replace(".png", "_watershed_mask.png")
        return Image.open(filename)

    def __getitem__(self, idx):
        image_size = (self.size, self.size)
        img = np.array(Image.open(self.img_files[idx]).convert('RGB').resize(image_size, resample=PIL.Image.Resampling.LANCZOS)).astype(np.uint8)
        msk = np.array(self._get_mask(idx).convert('L').resize(image_size, resample=PIL.Image.Resampling.NEAREST)).astype(np.uint8)
        img_msk = np.concatenate([img, msk[:,:, np.newaxis]], axis=-1)
        img_msk = (img_msk/127.5 - 1.0).astype(np.float32)
        return {'image': img_msk}