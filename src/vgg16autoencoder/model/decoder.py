import torch
from torch import nn
import toml

from vgg16common import LOGGER


class VGG16Decoder(nn.Module):
    def __init__(self, depth=5, use_gpu=False):
        LOGGER.info(f"Initializing decoder (depth: {depth}, use_gpu: {use_gpu})")
        super().__init__()
        self.depth = depth
        self.use_gpu = use_gpu
        all_layers = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            # nn.ReLU(inplace=True),
        )
        if use_gpu:
            all_layers = all_layers.to("cuda")

        # Hardcoded indices for each depth
        indices = {
            1: 21,
            2: 16,
            3: 9,
            4: 2,
            5: 0,
        }
        self.model = all_layers[indices[depth] :]
        LOGGER.debug(f"Decoder layers\n{self.model}")

    @classmethod
    def from_state_dict(cls, *args, path, **kwargs):
        LOGGER.info(f"Loading model from {path}")
        model = cls(*args, **kwargs)
        model.load_state_dict(torch.load(path))
        LOGGER.info("Finished loading model")
        return model

    def forward(self, x):
        return self.model(x)

    def save_model(self, path, data_dict=None):
        LOGGER.info(f"Saving model in {path}")
        self.eval()
        torch.save(self.state_dict(), path)

        # Store the data in a TOML file contained in the same
        # directory as the weights.
        if data_dict is not None:
            with open(f"{'.'.join(path.split('.')[:-1])}.toml", "w") as f:
                LOGGER.info(f"Saving data in {'.'.join(path.split('.')[:-1])}.toml")
                toml.dump(data_dict, f)


class VGG19Decoder(nn.Module):
    def __init__(self, depth=5, use_gpu=False):
        LOGGER.info(f"Initializing VGG19 decoder (depth: {depth}, use_gpu: {use_gpu})")
        self.depth = depth
        self.use_gpu = use_gpu
        super().__init__()
        all_layers = nn.Sequential(
            # Depth 5
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),
            # Depth 4
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 256, (3, 3)),
            nn.ReLU(),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            # Depth 3
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 128, (3, 3)),
            nn.ReLU(),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, (3, 3)),
            nn.ReLU(),
            # Depth 2
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 64, (3, 3)),
            nn.ReLU(),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),
            # Depth 1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 3, (3, 3)),
        )
        if use_gpu:
            all_layers = all_layers.to("cuda")

        # Hardcoded indices for each depth
        indices = {
            1: 40,
            2: 33,
            3: 26,
            4: 13,
            5: 0,
        }
        self.model = all_layers[indices[depth] :]
        LOGGER.debug(f"Decoder layers\n{self.model}")

    @classmethod
    def from_state_dict(cls, *args, path, **kwargs):
        LOGGER.info(f"Loading model from {path}")
        model = cls(*args, **kwargs)
        model.model.load_state_dict(torch.load(path))
        LOGGER.info("Finished loading model")
        return model

    def forward(self, x):
        return self.model(x)
