import torchvision
import torch
import os
import toml

from vgg16autoencoder.dataset import VGG16DecoderImageDataset
from vgg16autoencoder.model import VGG16Encoder, VGG16Decoder
from vgg16autoencoder.loss import VGG16DecoderLossFunction
from vgg16autoencoder.training import train_model, show_curves
from vgg16autoencoder.logger import LOGGER, setup_logger
from vgg16autoencoder.arg_parser import PARSER
from vgg16autoencoder import PATH_TO_WEIGHTS


args = PARSER.parse_args()
setup_logger(args.quiet, args.debug, args.verbose, args.log)


if args.train:
    LOGGER.info(f"Running in training mode with {int(args.epochs)} epochs and batch sizes {int(args.batch_size[0])}\
        for training and {int(args.batch_size[1])} for validation.")
    LEARNING_RATE = 5e-4
    BATCH_SIZE = int(args.batch_size[0])
    VALIDATION_BATCH_SIZE = int(args.batch_size[1])
    EPOCHS = int(args.epochs)
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
    vgg_encoder = VGG16Encoder(depth=args.depth, use_gpu=USE_GPU)

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

    if args.debug or args.debug is None:
        DEBUG = True
        if args.debug is None:
            reduced_size = 5000
        else:
            reduced_size = int(args.debug)

        LOGGER.debug(f"Creating reduced dataset with size {reduced_size}.")
        reduced_ds = VGG16DecoderImageDataset.from_dir(
            "data/test2017",
            encoder=vgg_encoder,
            transform=transform,
            use_gpu=USE_GPU,
            size_limit=reduced_size,
        )
    else:
        DEBUG = False


    # Loss function and model to train
    criterion = VGG16DecoderLossFunction(1, use_gpu=USE_GPU)

    if args.from_weights or args.from_weights is None:
        LOGGER.info("Initializing decoder from weights.")
        path = args.from_weights
        if path is None:
            LOGGER.info("Setting decoder weights to the best ones.")
            path = os.path.join(PATH_TO_WEIGHTS, f"best{vgg_encoder.depth}.pt")

        try:
            vgg_decoder = VGG16Decoder.from_state_dict(depth=args.depth, path=path, use_gpu=USE_GPU)
            with open(f"{'.'.join(path.split('.')[:-1])}.toml", "r") as f:
                start_curves = toml.load(f)["loss_evolution"]
        except FileNotFoundError as e:
            if path == os.path.join(PATH_TO_WEIGHTS, f"best{vgg_encoder.depth}.pt"):
                LOGGER.warning("Best model was not found (maybe it was deleted?). Initializing from a scratch.")
                vgg_decoder = VGG16Decoder(depth=args.depth, use_gpu=USE_GPU)
                start_curves = None
            else:
                raise e

    else:
        LOGGER.info("Initializing decoder from scratch.")
        vgg_decoder = VGG16Decoder(depth=args.depth, use_gpu=USE_GPU)
        start_curves = None


    # Flush the memory in cuda before running
    torch.cuda.empty_cache()

    if DEBUG:
        _, _, train_ds, val_ds = reduced_ds.split(args.train_split)
    else:
        _, _, train_ds, val_ds = transformed_ds.split(args.train_split)

    # Run the training
    curves = train_model(
        vgg_decoder,
        train_ds,
        val_ds,
        EPOCHS,
        criterion,
        BATCH_SIZE,
        VALIDATION_BATCH_SIZE,
        LEARNING_RATE,
        vgg_encoder,
        use_gpu=USE_GPU,
        save_weights=args.save_weights,
        start_curves=start_curves,
        never_save=args.never_save,
    )

    if args.show_curves:
        show_curves(curves)
