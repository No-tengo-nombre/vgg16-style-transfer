import torch
from torch import nn
from torchvision import models

from vgg16common import LOGGER


class VGG16Encoder(nn.Module):
    def __init__(self, depth=5, use_gpu=False):
        LOGGER.info(f"Initializing VGG16 encoder (depth: {depth}, use_gpu: {use_gpu})")
        self.depth = depth
        self.use_gpu = use_gpu
        super().__init__()
        all_layers = (
            models.vgg16(weights=models.VGG16_Weights.DEFAULT).eval().features
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
        self.model = all_layers[: indices[depth] + 1]
        LOGGER.debug(f"Encoder layers\n{self.model}")

    def forward(self, x):
        return self.model(x)


class VGG19Encoder(nn.Module):
    def __init__(self, depth=5, use_gpu=False):
        LOGGER.info(f"Initializing VGG19 encoder (depth: {depth}, use_gpu: {use_gpu})")
        self.depth = depth
        self.use_gpu = use_gpu
        super().__init__()
        all_layers = nn.Sequential(
            nn.Conv2d(3, 3, (1, 1)),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(3, 64, (3, 3)),
            nn.ReLU(),
            # Depth 1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 128, (3, 3)),
            nn.ReLU(),
            # Depth 2
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, (3, 3)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 256, (3, 3)),
            nn.ReLU(),
            # Depth 3
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 512, (3, 3)),
            nn.ReLU(),
            # Depth 4
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),
            # Depth 5
        )
        if use_gpu:
            all_layers = all_layers.to("cuda")

        # Hardcoded indices for each depth
        indices = {
            1: 3,
            2: 10,
            3: 17,
            4: 30,
            5: 43,
        }
        self.model = all_layers[: indices[depth] + 1]
        LOGGER.debug(f"Encoder layers\n{self.model}")

    @classmethod
    def from_state_dict(cls, *args, path, **kwargs):
        LOGGER.info(f"Loading model from {path}")
        model = cls(*args, **kwargs)
        model.model.load_state_dict(torch.load(path))
        LOGGER.info("Finished loading model")
        return model

    def forward(self, x):
        return self.model(x)
