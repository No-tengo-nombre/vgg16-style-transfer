import torch
from torch import nn
import matplotlib.pyplot as plt

from vgg16common import LOGGER


class VGG16DecoderLossFunction(nn.Module):
    def __init__(self, weight=1, show_progress=False, show_images=False, use_gpu=False):
        LOGGER.info("Initializing loss function.")
        super().__init__()
        self.weight = weight
        self.show_progress = show_progress
        self.show_images = show_images
        self.use_gpu = use_gpu

    def forward(self, recon_image, input_image, encoder):
        if self.use_gpu:
            encoder = encoder.cuda()
            input_image = input_image.cuda()
            recon_image = recon_image.cuda()

        input_features = encoder(input_image)
        recon_features = encoder(recon_image)

        image_loss = torch.pow(torch.linalg.norm(input_image - recon_image), 2)
        feature_loss = torch.pow(torch.linalg.norm(input_features - recon_features), 2)
        total_loss = image_loss + self.weight * feature_loss

        if self.show_progress:
            if self.show_images:
                for i in range(input_image.detach().shape[0]):
                    fig, ax = plt.subplots(1, 2)
                    ax[0].imshow(input_image.detach()[i].permute(1, 2, 0).cpu().numpy())
                    ax[1].imshow(
                        recon_image.detach()[i].permute(1, 2, 0).cpu().numpy()
                        / torch.max(recon_image.detach().cpu())
                    )

                    ax[0].set_title("Input image")
                    ax[1].set_title("Reconstructed image")
            print(
                f"""
                \rLoss = {total_loss}
                \r=========================
                \rInput                  -> {input_image.shape}
                \rInput min              -> {torch.min(input_image)}
                \rInput max              -> {torch.max(input_image)}
                \rReconstructed          -> {recon_image.shape}
                \rReconstructed min      -> {torch.min(recon_image)}
                \rReconstructed max      -> {torch.max(recon_image)}
                \rInput features         -> {input_features.shape}
                \rReconstructed features -> {recon_features.shape}
            """
            )
            plt.show()
        return total_loss
