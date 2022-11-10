import torch
import os
import numpy as np
from PIL import Image

from vgg16st.logger import LOGGER


class UnsupervisedImageDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, image_names, transform=None, use_gpu=False, size_limit=None):
        LOGGER.info("Initializing dataset.")
        self.img_dir = img_dir
        self.image_names = image_names[:size_limit]
        self.transform = transform
        self.use_gpu = use_gpu

    @classmethod
    def from_dir(cls, img_dir, *args, **kwargs):
        LOGGER.info("Initializing dataset from directory.")
        files = np.array([f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))])
        return cls(img_dir, files, *args, **kwargs)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.img_dir, self.image_names[idx])).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image