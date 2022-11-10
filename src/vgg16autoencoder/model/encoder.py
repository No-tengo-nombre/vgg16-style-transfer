from torch import nn
from torchvision import models

from vgg16autoencoder.logger import LOGGER


class VGG16Encoder(nn.Module):
    def __init__(self, depth=5, use_gpu=False):
        LOGGER.info(f"Initializing encoder (depth: {depth}, use_gpu: {use_gpu}).")
        self.depth = depth
        self.use_gpu = use_gpu
        super().__init__()
        all_layers = (
            models
            .vgg16(weights=models.VGG16_Weights.DEFAULT)
            .eval()
            .features
        )[:26]
        if use_gpu:
            all_layers = all_layers.to("cuda")

        # Hardcoded indices for each depth
        indices = {
            1: 4,
            2: 9,
            3: 16,
            4: 23,
            5: 26,
        }
        self.model = all_layers[:indices[depth] + 1]
        LOGGER.info(f"Encoder layers\n{self.model}.")

    def forward(self, x):
        return self.model(x)
