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
    LOGGER.info("Sending data to GPU.")
    if use_gpu:
        content_img = content_img.to("cuda")
        style_img = style_img.to("cuda")

    # Apply each stylization
    LOGGER.info("Applying stylization.")
    content = content_img
    for encoder, decoder in zip(encoders, decoders):
        LOGGER.info(f"Stylization level {encoder.depth}")

        # Encode
        content_feats = encoder(content)
        style_feats = encoder(style_img)
        LOGGER.info(f"Content feats {content_feats.shape}, style feats {style_feats.shape}.")

        # Stylize
        stylized_feats = wct(content_feats, style_feats)
        stylized_img = decoder(stylized_feats.reshape(1, *stylized_feats.shape))[0]
        content = stylized_img

    # Send images to the CPU
    content_img = content_img.detach().cpu()
    style_img = style_img.detach().cpu()
    content = inverse_normalization(content).detach().cpu()

    return content
