import os
import sys
import torch
import torchvision

from vgg16autoencoder import PATH_TO_WEIGHTS
from vgg16autoencoder.constants import NORM_MEAN, NORM_STD
from vgg16autoencoder.dataset import VGG16DecoderImageDataset
from vgg16autoencoder.loss import VGG16DecoderLossFunction
from vgg16autoencoder.model import VGG16Decoder, VGG16Encoder, VGG19Decoder, VGG19Encoder


def main():
    vgg16_encoder = VGG16Encoder(depth=5)
    vgg16_decoder = VGG16Decoder.from_state_dict(depth=5, path=os.path.join(PATH_TO_WEIGHTS, "best5.pt"))
    vgg19_encoder = VGG19Encoder.from_state_dict(depth=5, path=os.path.join(PATH_TO_WEIGHTS, "vgg19", "encoder5.pt"))
    vgg19_decoder = VGG19Decoder.from_state_dict(depth=5, path=os.path.join(PATH_TO_WEIGHTS, "vgg19", "decoder5.pt"))

    untransformed_val_ds = VGG16DecoderImageDataset.from_dir(
        "data/val2017",
        encoder=vgg16_encoder,
        transform=torchvision.transforms.Compose((
            torchvision.transforms.ToTensor(),
        )),
    )
    val_ds = VGG16DecoderImageDataset.from_dir(
        "data/val2017",
        encoder=vgg16_encoder,
        transform=torchvision.transforms.Compose(
            (
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize(256),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.Normalize(NORM_MEAN, NORM_STD),
            )
        )
    )

    loss_fn = VGG16DecoderLossFunction()
    idx = int(sys.argv[1])

    input_image = val_ds.get_image(idx)
    input_tensor = torch.zeros(1, *input_image.shape)
    input_tensor[0] = input_image

    feats_vgg16 = vgg16_encoder(input_tensor)
    feats_vgg19 = vgg19_encoder(input_tensor)
    recon_vgg16 = vgg16_decoder(feats_vgg16)
    recon_vgg19 = vgg19_decoder(feats_vgg19)


    loss_vgg16 = loss_fn(recon_vgg16, input_tensor, vgg16_encoder)
    loss_vgg19 = loss_fn(recon_vgg19, input_tensor, vgg19_encoder)
    print("Losses")
    print(f"VGG16: {loss_vgg16:.4e}")
    print(f"VGG19: {loss_vgg19:.4e}")


if __name__ == "__main__":
    main()
