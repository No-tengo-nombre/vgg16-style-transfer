import torchvision
import torch

from style_transfer.dataset import VGG16DecoderImageDataset
from style_transfer.model import VGG16Encoder, VGG16Decoder
from style_transfer.loss import VGG16DecoderLossFunction
from style_transfer.training import train_model, show_curves


LEARNING_RATE = 5e-4
BATCH_SIZE = 10
EPOCHS = 1
USE_GPU = True
NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD = (0.229, 0.224, 0.225)

# Define the transform for the data
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.Normalize(NORM_MEAN, NORM_STD),
])

vgg_encoder = VGG16Encoder(use_gpu=USE_GPU)

# Create the datasets
untransformed_ds = VGG16DecoderImageDataset.from_dir(
    "data/test2017",
    encoder=vgg_encoder,
    transform=torchvision.transforms.ToTensor(),
    use_gpu=USE_GPU,
)
transformed_ds = VGG16DecoderImageDataset.from_dir(
    "data/test2017",
    encoder=vgg_encoder,
    transform=transform,
    use_gpu=USE_GPU,
)
reduced_ds = VGG16DecoderImageDataset.from_dir(
    "data/test2017",
    encoder=vgg_encoder,
    transform=transform,
    use_gpu=USE_GPU,
    size_limit=1000,
)

# Loss function and model to train
criterion = VGG16DecoderLossFunction(1, use_gpu=USE_GPU)
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
)

show_curves(curves)
