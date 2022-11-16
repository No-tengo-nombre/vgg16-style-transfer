import matplotlib.pyplot as plt
import os
from PIL import Image
import re
import torchvision

from vgg16autoencoder.model import VGG16Decoder, VGG16Encoder
from vgg16autoencoder.constants import NORM_MEAN, NORM_STD
from vgg16autoencoder import PATH_TO_WEIGHTS
from vgg16st.transforms import WhiteningColoring


# Regex pattern for matching the user input
DEPTH_PATTERN = re.compile(r"([\+\-]?)(\d*)")
# TO_TENSOR = torchvision.transforms.ToTensor()
# NORMALIZATION = torchvision.transforms.Normalize(NORM_MEAN, NORM_STD),


def st_main(args):
    # Set up the transforms
    img_transform = torchvision.transforms.Compose(
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(NORM_MEAN, NORM_STD),
    )
    inverse_normalization = torchvision.transforms.Normalize(
        -NORM_MEAN / NORM_STD,
        1 / NORM_STD
    )
    wct = WhiteningColoring(args.alpha, method=args.method)

    # Determine the depths of the model
    depths = args.depth
    model_depths = []

    for d in depths:
        match DEPTH_PATTERN.search(d).groups():
            case ("-", num):
                model_depths.extend(range(1, num + 1))
            case ("+", num):
                model_depths.extend(range(num, 6))
            case ("", num):
                model_depths.append(num)

    # Create the encoders and decoders
    encoders = [VGG16Encoder(d) for d in model_depths]
    decoders = [
        VGG16Decoder.from_state_dict(
            depth=d,
            path=os.path.join(
                PATH_TO_WEIGHTS,
                f"best{d}.pt"
            )
        )
        for d in model_depths
    ]

    # Load the images
    content_img = img_transform(Image.open(args.content).convert("RGB"))
    style_img = img_transform(Image.open(args.style).convert("RGB"))

    # Load stuff into the GPU
    if args.gpu:
        for m in encoders:
            m.cuda()
        for m in decoders:
            m.cuda()

        content_img.to("cuda")
        style_img.to("cuda")

    # Apply each stylization
    content = content_img
    for encoder, decoder in zip(encoders, decoders):
        # Encode
        content_feats = encoder(content)
        style_feats = encoder(style_img)

        # Stylize
        stylized_feats = wct(content_feats, style_feats)
        stylized_img = decoder(stylized_feats)
        content = inverse_normalization(stylized_img)

    # Send images to the CPU
    content_img = content_img.cpu()
    style_img = style_img.cpu()
    content = content.cpu()

    # Image plotting
    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(content_img)
    ax[1].imshow(content)
    ax[2].imshow(style_img)

    ax[0].set_title("Original image")
    ax[1].set_title("Stylized image")
    ax[2].set_title("Style image")

    if args.save:
        fig.savefig(args.save, bbox_inches="tight")

    if args.plot:
        plt.show()
