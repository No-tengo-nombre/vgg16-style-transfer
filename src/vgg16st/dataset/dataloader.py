import torch
import numpy as np

from vgg16st.logger import LOGGER


class VGG16DecoderImageDataloader:
    def __init__(self, dataset, batch_size, use_gpu=False):
        LOGGER.info("Initializing dataloader.")
        self.dataset = dataset
        self.batch_size = batch_size
        self.use_gpu = use_gpu

    def __len__(self):
        return int(np.ceil(len(self.dataset) / self.batch_size))

    def __iter__(self):
        self.indices = list(range(len(self.dataset)))
        np.random.default_rng().shuffle(self.indices)
        return self

    def __next__(self):
        if not self.indices:
            raise StopIteration
        indices = self.indices[:self.batch_size]
        self.indices = self.indices[self.batch_size:]
        result_feats = torch.zeros((self.batch_size, 512, 14, 14))
        result_image = torch.zeros((self.batch_size, 3, 224, 224))
        for i, idx in enumerate(indices):
            features, image = self.dataset[idx]
            result_feats[i] = features
            result_image[i] = image
        if self.use_gpu:
            result_feats = result_feats.cuda()
            result_image = result_image.cuda()
        return result_feats, result_image
