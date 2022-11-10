import torch
from torch import nn
import toml

from vgg16st.logger import LOGGER


class VGG16Decoder(nn.Module):
    def __init__(self, depth=5, use_gpu=False):
        LOGGER.info("Initializing decoder.")
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
            1: 1,
            2: 8,
            3: 15,
            4: 20,
            5: 25,
        }
        self.model = all_layers[:indices[depth] + 1]

    @classmethod
    def from_state_dict(cls, *args, path, **kwargs):
        model = cls(*args, **kwargs)
        model.load_state_dict(torch.load(path))
        return model

    def forward(self, x):
        return self.model(x)

    def save_model(self, path, data_dict=None):
        LOGGER.info(f"Saving model in {path}.")
        self.eval()
        torch.save(self.state_dict(), path)

        # Store the data in a TOML file contained in the same
        # directory as the weights.
        if data_dict is not None:
            with open(f"{''.join(path.split('.')[:-1])}.toml", "w") as f:
                LOGGER.info(f"Saving data in {''.join(path.split('.')[:-1])}.toml.")
                toml.dump(data_dict, f)
