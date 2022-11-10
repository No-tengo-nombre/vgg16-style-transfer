import torchvision
import torch
import argparse
import os

from vgg16_autoencoder.dataset import VGG16DecoderImageDataset
from vgg16_autoencoder.model import VGG16Encoder, VGG16Decoder
from vgg16_autoencoder.loss import VGG16DecoderLossFunction
from vgg16_autoencoder.training import train_model, show_curves
from vgg16_autoencoder.logger import LOGGER, setup_logger
from vgg16_autoencoder import PATH_TO_WEIGHTS


desc_str = """VGG16 training and evaluation code."""

parser = argparse.ArgumentParser(description=desc_str)
parser.add_argument(
    "-T",
    "--train",
    action="store_true",
    help="Train the model.",
)
parser.add_argument(
    "-E",
    "--evaluate",
    action="store",
    help="Evaluate the model.",
)
parser.add_argument(
    "-d",
    "--debug",
    action="store_true",
    help="Run in debug mode.",
)
parser.add_argument(
    "--log",
    action="store",
    help="Generate log files for the current run in the given directory.",
)
parser.add_argument(
    "-q",
    "--quiet",
    action="store_true",
    help="Disable logging.",
)
parser.add_argument(
    "-v",
    "--verbose",
    action="store_true",
    help="Run in verbose mode.",
)
parser.add_argument(
    "-b",
    "--batch-size",
    action="store",
    default=1,
    help="Set the batch size.",
)
parser.add_argument(
    "-s",
    "--show-curves",
    action="store_true",
    help="Show the training curves.",
)
parser.add_argument(
    "-S",
    "--save-weights",
    action="store_true",
    help="Save the model's weights.",
)
parser.add_argument(
    "-w",
    "--from-weights",
    action="store",
    nargs="?",
    default=False,
    help="Load the model from weights.",
)

args = parser.parse_args()
setup_logger(args.quiet, args.debug, args.verbose, args.log)


if args.train:
    LOGGER.info(f"Running in training mode with batch size {int(args.batch_size)}.")
    LEARNING_RATE = 5e-4
    BATCH_SIZE = int(args.batch_size)
    EPOCHS = 1
    USE_GPU = True
    NORM_MEAN = (0.485, 0.456, 0.406)
    NORM_STD = (0.229, 0.224, 0.225)

    # Define the transform for the data
    LOGGER.info("Setting up transforms for dataset.")
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.Normalize(NORM_MEAN, NORM_STD),
    ])

    LOGGER.info("Creating encoder.")
    vgg_encoder = VGG16Encoder(use_gpu=USE_GPU)

    LOGGER.info("Creating datasets.")
    LOGGER.debug("Creating untransformed dataset.")
    # Create the datasets
    untransformed_ds = VGG16DecoderImageDataset.from_dir(
        "data/test2017",
        encoder=vgg_encoder,
        transform=torchvision.transforms.ToTensor(),
        use_gpu=USE_GPU,
    )
    LOGGER.debug("Creating transformed dataset.")
    transformed_ds = VGG16DecoderImageDataset.from_dir(
        "data/test2017",
        encoder=vgg_encoder,
        transform=transform,
        use_gpu=USE_GPU,
    )
    LOGGER.debug("Creating reduced dataset.")
    reduced_ds = VGG16DecoderImageDataset.from_dir(
        "data/test2017",
        encoder=vgg_encoder,
        transform=transform,
        use_gpu=USE_GPU,
        size_limit=10,
    )

    # Loss function and model to train
    criterion = VGG16DecoderLossFunction(1, use_gpu=USE_GPU)

    if args.from_weights or args.from_weights is None:
        LOGGER.info("Initializing decoder from weights.")
        path = args.from_weights
        if path is None:
            LOGGER.info("Setting decoder weights to the best ones.")
            path = os.path.join(PATH_TO_WEIGHTS, "best.pt")
        vgg_decoder = VGG16Decoder.from_state_dict(path=path, use_gpu=USE_GPU)
    else:
        LOGGER.info("Initializing decoder from scratch.")
        vgg_decoder = VGG16Decoder(use_gpu=USE_GPU)


    # Flush the memory in cuda before running
    torch.cuda.empty_cache()

    _, _, train_ds, val_ds = reduced_ds.split(0.7)

    # Run the training
    curves = train_model(
        vgg_decoder,
        train_ds,
        val_ds,
        EPOCHS,
        criterion,
        BATCH_SIZE,
        LEARNING_RATE,
        vgg_encoder,
        use_gpu=USE_GPU,
        save_weights=args.save_weights,
    )

    if args.show_curves:
        show_curves(curves)
