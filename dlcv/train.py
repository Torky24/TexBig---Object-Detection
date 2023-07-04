from dlcv.detection.engine import train_one_epoch, evaluate
import torch
from torch.utils.data import DataLoader


def train_model(
    model: torch.nn.Module,
    train_data_loader: DataLoader,
    validation_data_loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
) -> None:
    """
    Trains a given model for a specified number of epochs.

    Args:
        model (torch.nn.Module): The model to train.
        train_data_loader (DataLoader): The data loader for the training data.
        validation_data_loader (DataLoader): The data loader for the validation data.
        device (torch.device): The device to run the training on (e.g., "cuda", "cpu").
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        num_epochs (int): The number of epochs to train the model.

    Returns:
        None
    """
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(
            model, optimizer, train_data_loader, device, epoch, print_freq=1
        )

        print(f"Epoch [{epoch}]: Train Loss: {train_loss}")

        torch.save(model.state_dict(), f"trained_model_epoch{epoch + 1}.pth")

        val_loss, mAP = evaluate(model, validation_data_loader, device)
        print(f"Epoch [{epoch}]: Validation Loss: {val_loss}")
        print(f"mAP for {epoch+1} epoch is {mAP}")
