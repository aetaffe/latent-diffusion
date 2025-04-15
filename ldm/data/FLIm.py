import PIL
from torch.utils.data import Dataset
import albumentations
import numpy as np
from PIL import Image
import supervision as sv
import json
from pathlib import Path
import cv2

class ImagePaths(Dataset):
    def __init__(self, paths, size=None, use_augmentations=False, aug_p=0.2, labels=None):
        self.size = size
        self.use_augmentations = use_augmentations
        self.labels = dict() if labels is None else labels
        self.labels["file_path_"] = paths
        self._length = len(paths)

        if self.size is not None and self.size > 0:
            augmentations = []
            if self.use_augmentations:
                augmentations.extend([
                    albumentations.RandomSizedCrop(
                        min_max_height=(self.size//2, self.size//2),
                        size=(self.size, self.size),
                        interpolation=cv2.INTER_LANCZOS4,
                        mask_interpolation=cv2.INTER_NEAREST),
                    albumentations.HorizontalFlip(p=aug_p),
                    albumentations.Rotate(limit=(-90,90), p=aug_p),
                    albumentations.ColorJitter(p=aug_p),
                ])

            self.preprocessor = albumentations.Compose(augmentations)
        else:
            self.preprocessor = lambda **kwargs: kwargs

    def __len__(self):
        return self._length

    @staticmethod
    def _polygon_to_mask(polygons, width, height):
        mask = np.zeros((height, width))
        for shape in polygons:
            if shape['label'] == 'AimingBeam':
                points = np.int32(shape['points'])
                instr_mask = sv.polygon_to_mask(np.array(points), (width, height))
                instr_mask = np.where(instr_mask == 1, 255, 0)
                mask += instr_mask
        return mask

    def _get_segmentation_mask(self, fname):
        mask_json_fname = fname.replace('.jpg', '.json')
        with open(mask_json_fname) as f:
            mask_json = json.load(f)
            mask = self._polygon_to_mask(mask_json['shapes'], mask_json['imageWidth'], mask_json['imageHeight']).astype(np.uint8)
        return mask

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        seg_mask = self._get_segmentation_mask(image_path)
        if seg_mask is None:
            raise IOError(f'No seg_mask found for image file: {image_path}')
        elif not image.mode == "RGB":
            image = image.convert("RGB")

        image = image.resize((self.size, self.size), resample=Image.Resampling.LANCZOS)
        image = np.array(image).astype(np.uint8)
        seg_mask = np.array(Image.fromarray(seg_mask)
                            .convert("L")
                            .resize((self.size, self.size),
                                    resample=PIL.Image.Resampling.NEAREST))
        processed = self.preprocessor(image=image, mask=seg_mask)
        image = processed['image']
        seg_mask = processed['mask']
        image = np.concatenate([image, seg_mask[:,:,np.newaxis]],axis=2)
        image = (image/127.5 - 1.0).astype(np.float32)
        return image

    def __getitem__(self, i):
        example = dict()
        example["image"] = self.preprocess_image(self.labels["file_path_"][i])
        # for k in self.labels:
        #     example[k] = self.labels[k][i]
        return example


class DatasetBase(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data = None
        self.keys = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example = self.data[i]
        ex = {}
        if self.keys is not None:
            for k in self.keys:
                ex[k] = example[k]
        else:
            ex = example
        return ex

class FLImTrain(DatasetBase):
    def __init__(self, size, root_dir, keys=None, use_augmentations=False, aug_p=0.2):
        super().__init__()
        paths = [str(path) for path in list(Path(root_dir).rglob('*.jpg'))]
        self.data = ImagePaths(paths=paths, size=size, use_augmentations=use_augmentations, aug_p=aug_p)
        self.keys = keys