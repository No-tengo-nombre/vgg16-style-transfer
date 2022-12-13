import torch
import numpy as np
from PIL import Image
import os

from .unsupervised import UnsupervisedImageDataset
from vgg16common import LOGGER


class VGG16DecoderImageDataset(UnsupervisedImageDataset):
    def __init__(self, *args, encoder, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = encoder

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.img_dir, self.image_names[idx])).convert(
            "RGB"
        )
        if self.transform is not None:
            image = self.transform(image)

        if self.use_gpu:
            image = image.to("cuda")
        features = self.encoder(image)
        return features, image

    def split(self, train_size):
        LOGGER.info(f"Splitting dataset with train size {train_size}")
        total_size = len(self.image_names)
        train_idx = np.random.default_rng().choice(
            total_size, int(total_size * train_size), replace=False
        )
        test_idx = np.delete(np.arange(total_size), train_idx)

        train_ds = VGG16DecoderImageDataset(
            self.img_dir,
            self.image_names[train_idx],
            self.transform,
            self.use_gpu,
            encoder=self.encoder,
        )
        test_ds = VGG16DecoderImageDataset(
            self.img_dir,
            self.image_names[test_idx],
            self.transform,
            self.use_gpu,
            encoder=self.encoder,
        )
        return train_idx, test_idx, train_ds, test_ds
