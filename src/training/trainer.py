"""
Training and Validation Engine for Urban Sound Classification

Provides:
- train_one_epoch: executes single training epoch with CrossEntropyLoss and Adam/AdamW
- validate: calculates loss, top-1 accuracy, and returns predictions & ground truth
- train_model: complete training loop with model checkpointing and history tracking
- calculate_class_weights: handles dataset class imbalance via inverse frequency weighting

Usage:
    from src.training.trainer import train_model
    history = train_model(model, train_loader, val_loader, num_epochs=15, save_path="models/cnn_urban_sound.pth")
"""

from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Optional, Any
from collections import Counter
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.config.settings import settings

logger = logging.getLogger(__name__)


def calculate_class_weights(dataset: Any, num_classes: int = 10) -> torch.Tensor:
    """Calculate balanced inverse class frequency weights to handle dataset imbalance.

    Weight for class c: w_c = N / (K * N_c)
    where N is total samples, K is number of classes, and N_c is count of class c.

    Args:
        dataset: PyTorch Dataset (having .samples attribute or iterable).
        num_classes: Number of distinct classes.

    Returns:
        torch.FloatTensor of shape (num_classes,) with normalized weights.
    """
    counts = Counter()
    if hasattr(dataset, "samples"):
        for item in dataset.samples:
            counts[item["class_id"]] += 1
    else:
        # Fallback inspection of targets
        for _, label in dataset:
            cid = label.item() if isinstance(label, torch.Tensor) else int(label)
            counts[cid] += 1

    total_samples = sum(counts.values()) if counts else 1
    weights = torch.ones(num_classes, dtype=torch.float32)

    if total_samples > 0:
        for cid in range(num_classes):
            c_count = counts.get(cid, 0)
            if c_count > 0:
                weights[cid] = total_samples / (num_classes * c_count)
            else:
                weights[cid] = 1.0

        # Normalize weights so mean is 1.0
        weights = weights / weights.mean()

    return weights


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
    class_weights: Optional[torch.Tensor] = None,
    early_stopping_patience: Optional[int] = None,
    training_data_type: Optional[str] = None,
    is_synthetic: Optional[bool] = None,
    device: Optional[torch.device] = None,
    save_path: Optional[str | Path] = None,
    overwrite: bool = True,
    verbose: bool = True,
) -> dict[str, list[float]]:
    """Complete training and validation loop with rich provenance checkpointing.

    Args:
        model: PyTorch model to train.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        num_epochs: Number of epochs to train.
        lr: Initial learning rate.
        weight_decay: L2 regularization factor.
        class_weights: Optional class weighting tensor for CrossEntropyLoss.
        training_data_type: Provenance string ("urbansound8k" or "synthetic").
        is_synthetic: Boolean indicating if synthetic data was used.
        device: Device to train on (auto-detects CUDA/CPU if None).
        save_path: Path where best checkpoint .pth will be saved.
        overwrite: Whether to permit overwriting an existing checkpoint.
        verbose: Whether to log epoch results.

    Returns:
        Dictionary with training history (train_loss, train_acc, val_loss, val_acc).

    Raises:
        FileExistsError: If save_path exists and overwrite is False.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Resolve training data provenance
    if is_synthetic is None:
        is_synthetic = getattr(train_loader.dataset, "is_synthetic", False)
    if training_data_type is None:
        training_data_type = getattr(
            train_loader.dataset,
            "training_data_type",
            "synthetic" if is_synthetic else "urbansound8k",
        )
    training_data_type = str(training_data_type).lower()

    save_path = Path(save_path or settings.DEFAULT_MODEL_PATH)
    if save_path.exists() and not overwrite:
        raise FileExistsError(
            f"Checkpoint already exists at '{save_path}'. Pass overwrite=True to allow overwriting."
        )
    save_path.parent.mkdir(parents=True, exist_ok=True)

    model.to(device)

    criterion_weights = class_weights.to(device) if class_weights is not None else None
    criterion = nn.CrossEntropyLoss(weight=criterion_weights)
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
    epochs_no_improve = 0

    if verbose:
        logger.info(
            f"Starting training on device: {device} | Total Epochs: {num_epochs} | "
            f"Data: {training_data_type} (is_synthetic={is_synthetic})"
        )

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
            epochs_no_improve = 0
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_acc": best_val_acc,
                "val_loss": val_loss,
                "architecture": "UrbanSoundCNN",
                "num_classes": settings.NUM_CLASSES,
                "classes": settings.URBAN_SOUND_CLASSES,
                "input_shape": [1, settings.N_MELS, 173],
                "preprocessing": {
                    "sample_rate": settings.SAMPLE_RATE,
                    "duration": settings.AUDIO_DURATION,
                    "n_fft": settings.N_FFT,
                    "hop_length": settings.HOP_LENGTH,
                    "n_mels": settings.N_MELS,
                    "ref": "np.max",
                    "top_db": 80.0,
                },
                "training_data_type": training_data_type,
                "is_synthetic": bool(is_synthetic),
                "is_trained": not is_synthetic,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            torch.save(checkpoint, str(save_path))
            if verbose:
                print(f"  --> Saved new best checkpoint to {save_path} (Val Acc: {best_val_acc:.2f}%)")
        else:
            epochs_no_improve += 1
            if early_stopping_patience is not None and epochs_no_improve >= early_stopping_patience:
                if verbose:
                    print(
                        f"Early stopping triggered: no validation improvement for "
                        f"{early_stopping_patience} consecutive epochs. Best Val Acc: {best_val_acc:.2f}%."
                    )
                break

    return history
