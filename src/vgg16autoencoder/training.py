import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import numpy as np
from datetime import datetime
import os
import toml

from .dataset import VGG16DecoderImageDataloader
from vgg16common import LOGGER
from vgg16autoencoder import PATH_TO_WEIGHTS


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
                batch_size, validation_batch_size, lr, encoder, use_gpu=False, loader_kwargs=None,
                save_weights=True, start_curves=None, never_save=False, start_epoch=0):
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
        val_dataset, batch_size=validation_batch_size, use_gpu=use_gpu, **loader_kwargs,
    )

    # If start_curves is given it should be a dictionary with keys "train_loss"
    # and "val_loss".
    if start_curves is None:
        curves = {
            "training": [],
            "validation": [],
        }
        train_loss = np.inf
        val_loss = np.inf
    else:
        curves = start_curves
        train_loss = start_curves["training"][-1]
        val_loss = start_curves["validation"][-1]

    n_batches = len(train_loader)

    # Initialize the values to display them

    LOGGER.info("Starting training.")
    for epoch in range(epochs):
        # Initialize the optimizer per epoch
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Metrics
        cumulative_train_loss = 0
        train_loss_count = 0

        # Train the model
        model.train()
        progress_bar = tqdm(
            train_loader,
            total=n_batches,
            desc=f"({epoch + 1 + start_epoch}/{epochs + start_epoch}) Losses | Train {train_loss:.4e} - Val {val_loss:.4e}",
        )
        for x_batch, y_batch in progress_bar:
            if use_gpu:
                x_batch = x_batch.cuda()
                y_batch = y_batch.cuda()

            _, loss = train_step(
                x_batch, y_batch, model, optimizer, criterion, use_gpu, encoder,
            )
            cumulative_train_loss += loss.item()
            train_loss_count += 1

            train_loss = cumulative_train_loss / train_loss_count
            progress_bar.desc = f"({epoch + 1 + start_epoch}/{epochs + start_epoch}) Losses | Train {train_loss:.4e} - Val {val_loss:.4e}"

        train_loss = cumulative_train_loss / train_loss_count

        # Run the evaluation
        del x_batch, y_batch
        torch.cuda.empty_cache()
        val_loss = evaluate(val_loader, model, encoder, criterion, use_gpu)

        # Save the curves
        curves["training"].append(train_loss)
        curves["validation"].append(val_loss)

        # Save the model
        if not never_save:
            data_dict = {
                "parameters": {
                    "current_epoch": epoch + 1 + start_epoch,
                    "epochs": epochs,
                    "criterion": criterion.__class__.__name__,
                    "batch_size": batch_size,
                    "learning_rate": lr,
                    "use_gpu": use_gpu,
                },
                "final_losses": {
                    "training": train_loss,
                    "validation": val_loss,
                },
                "loss_evolution": {
                    "training": curves["training"],
                    "validation": curves["validation"],
                }
            }
            if save_weights:
                # now = datetime.now()
                # file_name = os.path.join(PATH_TO_WEIGHTS, f"{now.year}{now.month:02}{now.day:02}_{now.hour:02}{now.minute:02}{now.second:02}.pt")
                file_name = os.path.join(PATH_TO_WEIGHTS, save_weights)
                model.save_model(file_name, data_dict)

            # Save it as best model if it is appropriate
            try:
                with open(os.path.join(PATH_TO_WEIGHTS, f"best{model.depth}.toml"), "r") as f:
                    best_model = toml.load(f)
                if float(best_model["final_losses"]["validation"]) > val_loss:
                    LOGGER.warning("Saving new best model.")
                    model.save_model(os.path.join(PATH_TO_WEIGHTS, f"best{model.depth}.pt"), data_dict)
            except FileNotFoundError:
                LOGGER.warning("Best model not found (maybe it was deleted?). Saving it anyways.")
                model.save_model(os.path.join(PATH_TO_WEIGHTS, f"best{model.depth}.pt"), data_dict)


    model = model.cpu()

    return curves


def show_curves(curves):
    LOGGER.info("Showing training curves.")
    fig, ax = plt.subplots(1, 1, figsize=(13, 5))
    fig.set_facecolor("white")

    epochs = np.arange(len(curves["validation"])) + 1

    ax.plot(epochs, curves["validation"], label="validation")
    ax.plot(epochs, curves["training"], label="training")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss evolution during training")
    ax.legend()

    plt.show()
