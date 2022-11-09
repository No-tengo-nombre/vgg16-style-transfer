import matplotlib.pyplot as plt
import time
import torch
from tqdm import tqdm
import numpy as np

from .dataset import VGG16DecoderImageDataloader
from vgg16_autoencoder.logger import LOGGER


def train_step(x_batch, y_batch, model, optimizer, criterion, use_gpu, encoder):
    y_predicted = model(x_batch)
    if use_gpu:
        y_predicted = y_predicted.cuda()
    loss = criterion(y_predicted, y_batch, encoder)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return y_predicted, loss


def evaluate(val_loader, model, encoder, criterion, use_gpu):
    LOGGER.info("Evaluating model.")
    cumulative_loss = 0

    for x_val, y_val in val_loader:
        if use_gpu:
            x_val = x_val.cuda()
            y_val = y_val.cuda()

        y_predicted = model(x_val)
        loss = criterion(y_predicted, y_val, encoder)

        cumulative_loss += loss.item()

    LOGGER.debug(f"Validation loss: {cumulative_loss / len(val_loader):.4f}.")
    return cumulative_loss / len(val_loader)


def train_model(model, train_dataset, val_dataset, epochs, criterion,
                batch_size, lr, encoder, n_evaluations_per_epoch=6,
                use_gpu=False, loader_kwargs=None):
    LOGGER.info("Training model.")
    if use_gpu:
        model = model.cuda()
        encoder = encoder.cuda()
        criterion = criterion.cuda()
    if loader_kwargs is None:
        loader_kwargs = {}

    # Dataloaders
    LOGGER.info("Generating dataloaders.")
    train_loader = VGG16DecoderImageDataloader(
        train_dataset, batch_size=batch_size, use_gpu=use_gpu, **loader_kwargs,
    )
    val_loader = VGG16DecoderImageDataloader(
        val_dataset, batch_size=batch_size, use_gpu=use_gpu, **loader_kwargs,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    curves = {
        "train_loss": [],
        "val_loss": [],
    }

    initial_time = time.perf_counter()
    n_batches = len(train_loader)

    # Initialize the values to display them
    train_loss = None
    val_loss = None

    LOGGER.info("Starting training.")
    for epoch in range(epochs):
        # Metrics
        cumulative_train_loss = 0
        train_loss_count = 0

        # Train the model
        model.train()
        progress_bar = tqdm(
            enumerate(train_loader),
            total=n_batches,
            desc=f"EPOCH {epoch + 1}/{epochs} | Training the decoder (train loss = {train_loss}, validation loss = {val_loss})",
        )
        for i, (x_batch, y_batch) in progress_bar:
            if use_gpu:
                x_batch = x_batch.cuda()
                y_batch = y_batch.cuda()

            y_predicted, loss = train_step(
                x_batch, y_batch, model, optimizer, criterion, use_gpu, encoder,
            )
            cumulative_train_loss += loss.item()
            train_loss_count += 1

            train_loss = cumulative_train_loss / train_loss_count
            progress_bar.desc = f"EPOCH {epoch + 1}/{epochs} | Training the decoder (train loss = {train_loss}, validation loss = {val_loss})"

        train_loss = cumulative_train_loss / train_loss_count

        # Run the evaluation
        del x_batch, y_batch
        torch.cuda.empty_cache()
        val_loss = evaluate(val_loader, model, encoder, criterion, use_gpu)

        # Save the curves
        curves["train_loss"].append(train_loss)
        curves["val_loss"].append(val_loss)

    model = model.cpu()
    return curves


def show_curves(curves):
    LOGGER.info("Showing training curves.")
    fig, ax = plt.subplots(1, 1, figsize=(13, 5))
    fig.set_facecolor("white")

    epochs = np.arange(len(curves["val_loss"])) + 1

    ax.plot(epochs, curves["val_loss"], label="validation")
    ax.plot(epochs, curves["train_loss"], label="training")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss evolution during training")
    ax.legend()

    plt.show()
