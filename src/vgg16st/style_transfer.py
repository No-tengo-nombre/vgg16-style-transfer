import torch
import torchvision
import os

from vgg16autoencoder.model import VGG16Decoder, VGG16Encoder
from vgg16autoencoder import PATH_TO_WEIGHTS
from vgg16common import LOGGER, NORM_MEAN, NORM_STD
from vgg16st.transforms import WhiteningColoring


def transfer_style(content, style, depths=(1, 2, 3, 4, 5), use_gpu=False, alpha=1, method="paper"):
    normalization = torchvision.transforms.Normalize(NORM_MEAN, NORM_STD)
    inverse_normalization = torchvision.transforms.Normalize(
        -NORM_MEAN / NORM_STD,
        1 / NORM_STD
    )
    wct = WhiteningColoring(alpha, method=method)

    # Create the encoders and decoders
    encoders = []
    decoders = []
    LOGGER.info("Creating encoders and decoders")
    for d in depths:
        encoders.append(VGG16Encoder(depth=d, use_gpu=use_gpu))
        decoders.append(
            VGG16Decoder.from_state_dict(
                depth=d,
                path=os.path.join(
                    PATH_TO_WEIGHTS,
                    f"best{d}.pt",
                ),
                use_gpu=use_gpu,
            )
        )

    # Normalizing images
    content_img = normalization(content)
    style_img = normalization(style)

    # Load stuff into the GPU
    if use_gpu:
        LOGGER.info("Sending data to GPU.")
        content_img = content_img.to("cuda")
        style_img = style_img.to("cuda")

    # Apply each stylization
    LOGGER.info("Applying stylization.")
    for encoder, decoder in zip(encoders, decoders):
        LOGGER.info(f"Stylization level {encoder.depth}")
        stylized_feats = wct(encoder(content_img), encoder(style_img))

        # Apply the decoder
        LOGGER.info("Decoding styled features.")
        content_img = decoder(stylized_feats.reshape(1, *stylized_feats.shape))[0]

    return inverse_normalization(content_img).detach().cpu()
