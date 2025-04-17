from pathlib import Path

import PIL
import albumentations
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset


class Cholec8kDataset(Dataset):
    def __init__(self, root_dir, size, use_augmentations=False, aug_p=0.2):
        self.img_files = [str(file) for file in list(Path(root_dir).rglob("*[!mask].png"))]
        self.size = size
        self.use_augmentations = use_augmentations
        if self.size is not None and self.size > 0:
            augmentations = []
            if self.use_augmentations:
                print(
                    f'Using RandomeSizedCrop, HorizontalFlip, Rotate, Color Jitter augmentations with augment percentage: {aug_p}')
                augmentations.extend([
                    albumentations.RandomSizedCrop(
                        min_max_height=(self.size//2, self.size//2),
                        size=(self.size, self.size),
                        interpolation=cv2.INTER_LANCZOS4,
                        mask_interpolation=cv2.INTER_NEAREST,
                        p=aug_p),
                    albumentations.HorizontalFlip(p=aug_p),
                    albumentations.Rotate(limit=(-90,90), p=aug_p),
                    albumentations.ColorJitter(p=aug_p),
                ])

            self.preprocessor = albumentations.Compose(augmentations)
        else:
            self.preprocessor = lambda **kwargs: kwargs

    def __len__(self):
        return len(self.img_files)

    def _get_mask(self, idx):
        filename = self.img_files[idx].replace(".png", "_watershed_mask.png")
        return Image.open(filename)

    def __getitem__(self, idx):
        image_size = (self.size, self.size)
        img = np.array(Image.open(self.img_files[idx]).convert('RGB').resize(image_size, resample=PIL.Image.Resampling.LANCZOS)).astype(np.uint8)
        msk = np.array(self._get_mask(idx).convert('L').resize(image_size, resample=PIL.Image.Resampling.NEAREST)).astype(np.uint8)
        processed = self.preprocessor(image=img, mask=msk)
        img = processed['image']
        msk = processed['mask']
        img_msk = np.concatenate([img, msk[:,:, np.newaxis]], axis=-1)
        img_msk = (img_msk/127.5 - 1.0).astype(np.float32)
        return {'image': img_msk}