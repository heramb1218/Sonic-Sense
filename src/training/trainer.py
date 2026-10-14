"""
Training and Validation Engine for Urban Sound Classification

Provides:
- train_one_epoch: executes single training epoch with CrossEntropyLoss and Adam/AdamW
- validate: calculates loss, top-1 accuracy, and returns predictions & ground truth
- train_model: complete training loop with model checkpointing and history tracking

Usage:
    from src.training.trainer import train_model
    history = train_model(model, train_loader, val_loader, num_epochs=15, save_path="models/cnn_urban_sound.pth")
"""

import logging
from pathlib import Path
from typing import Optional, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config.settings import settings

logger = logging.getLogger(__name__)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Train the model for one epoch.

    Returns:
        Tuple of (average_loss, accuracy_percentage).
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = (100.0 * correct / total) if total > 0 else 0.0
    return epoch_loss, epoch_acc


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, list[int], list[int]]:
    """Evaluate the model on a validation dataset.

    Returns:
        Tuple of (val_loss, val_acc, all_preds, all_targets).
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            all_preds.extend(predicted.cpu().numpy().tolist())
            all_targets.extend(targets.cpu().numpy().tolist())

    val_loss = running_loss / total if total > 0 else 0.0
    val_acc = (100.0 * correct / total) if total > 0 else 0.0
    return val_loss, val_acc, all_preds, all_targets


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 20,
    lr: float = 0.001,
    weight_decay: float = 1e-4,
    device: Optional[torch.device] = None,
    save_path: Optional[str | Path] = None,
    verbose: bool = True,
) -> dict[str, list[float]]:
    """Complete training and validation loop with checkpointing.

    Args:
        model: PyTorch model to train.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        num_epochs: Number of epochs to train.
        lr: Initial learning rate.
        weight_decay: L2 regularization factor.
        device: Device to train on (auto-detects CUDA/CPU if None).
        save_path: Path where best checkpoint .pth will be saved.
        verbose: Whether to log epoch results.

    Returns:
        Dictionary with training history (train_loss, train_acc, val_loss, val_acc).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Learning rate scheduler (reduces LR on plateau)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_acc = -1.0
    save_path = Path(save_path or settings.DEFAULT_MODEL_PATH)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if verbose:
        logger.info(f"Starting training on device: {device} | Total Epochs: {num_epochs}")

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)

        scheduler.step(val_acc)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if verbose:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch [{epoch:02d}/{num_epochs:02d}] "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
                f"LR: {current_lr:.6f}"
            )

        # Save best model checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_acc": best_val_acc,
                "val_loss": val_loss,
                "classes": settings.URBAN_SOUND_CLASSES,
                "num_classes": settings.NUM_CLASSES,
            }
            torch.save(checkpoint, str(save_path))
            if verbose:
                print(f"  --> Saved new best checkpoint to {save_path} (Val Acc: {best_val_acc:.2f}%)")

    return history
