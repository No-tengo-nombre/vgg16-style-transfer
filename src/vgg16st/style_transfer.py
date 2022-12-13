import torchvision
from tqdm import tqdm
import os

from vgg16autoencoder.model import (
    VGG16Decoder,
    VGG16Encoder,
    VGG19Decoder,
    VGG19Encoder,
)
from vgg16autoencoder import PATH_TO_WEIGHTS
from vgg16common import LOGGER, NORM_MEAN, NORM_STD, DEFAULT_ST_SQUARE_SIZE
from vgg16st.exceptions import ModelException
from vgg16st.vendor import wct


def transfer_style(
    content,
    style,
    depths=(1, 2, 3, 4, 5),
    use_gpu=False,
    alpha=1,
    method="paper",
    model="vgg16",
    square_size=DEFAULT_ST_SQUARE_SIZE,
):
    RESIZE_SHAPE = (square_size, square_size)
    content_shape = content.shape[-2:]

    eval_resize = torchvision.transforms.Resize(RESIZE_SHAPE)
    content_resize = torchvision.transforms.Resize(content_shape)

    normalization = torchvision.transforms.Normalize(NORM_MEAN, NORM_STD)
    inverse_normalization = torchvision.transforms.Normalize(
        -NORM_MEAN / NORM_STD, 1 / NORM_STD
    )

    # Normalizing images
    content_img = eval_resize(normalization(content))
    style_img = eval_resize(normalization(style))

    # Load stuff into the GPU
    if use_gpu:
        LOGGER.info("Sending data to GPU")
        content_img = content_img.to("cuda")
        style_img = style_img.to("cuda")

    # Apply each stylization
    LOGGER.info("Applying stylization")
    for d in tqdm(depths, "Transferring style"):
        if model == "vgg16":
            encoder = VGG16Encoder(depth=d, use_gpu=use_gpu)
            decoder = VGG16Decoder.from_state_dict(
                depth=d,
                path=os.path.join(
                    PATH_TO_WEIGHTS,
                    f"best{d}.pt",
                ),
                use_gpu=use_gpu,
            )
        elif model == "vgg19":
            encoder = VGG19Encoder.from_state_dict(
                depth=d,
                path=os.path.join(
                    PATH_TO_WEIGHTS,
                    "vgg19",
                    f"encoder{d}.pt",
                ),
                use_gpu=use_gpu,
            )
            decoder = VGG19Decoder.from_state_dict(
                depth=d,
                path=os.path.join(
                    PATH_TO_WEIGHTS,
                    "vgg19",
                    f"decoder{d}.pt",
                ),
                use_gpu=use_gpu,
            )
        else:
            raise ModelException(f"Model {model} could not be found.")
        LOGGER.info(f"Stylization level {d}")
        stylized_feats = wct(
            alpha, encoder(content_img).detach(), encoder(style_img).detach()
        )

        # Apply the decoder
        LOGGER.info("Decoding styled features")
        content_img = decoder(stylized_feats)[0]

    return content_resize(inverse_normalization(content_img).detach()).cpu()
