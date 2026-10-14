"""
CNN Training Script — Phase 2

Trains the UrbanSoundCNN audio classifier on UrbanSound8K 10-fold dataset
or in synthetic mode for rapid local development/testing.

Usage:
    # Quick synthetic smoke test (runs 3 epochs on 100 samples)
    python scripts/train_cnn.py --synthetic --epochs 3

    # Standard training on UrbanSound8K folds 1-8 (val on fold 9)
    python scripts/train_cnn.py --epochs 20 --augment
"""

import sys
import argparse
import logging
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import settings
from src.models.cnn import UrbanSoundCNN
from src.training.dataset import UrbanSoundDataset, create_dataloaders
from src.training.trainer import train_model

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train PyTorch CNN for Urban Sound Classification.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--augment", action="store_true", help="Enable audio data augmentation during training")
    parser.add_argument("--synthetic", action="store_true", help="Force synthetic dataset mode for fast smoke test")
    parser.add_argument("--save-path", type=str, default=str(settings.DEFAULT_MODEL_PATH), help="Destination model path")
    args = parser.parse_args()

    print("=" * 60)
    print("URBAN SOUND CNN TRAINING")
    print("=" * 60)
    print(f"  Epochs:        {args.epochs}")
    print(f"  Batch Size:    {args.batch_size}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Augmentation:  {args.augment}")
    print(f"  Synthetic:     {args.synthetic}")
    print(f"  Save Path:     {args.save_path}")
    print("-" * 60)

    # Prepare datasets
    if args.synthetic:
        train_ds = UrbanSoundDataset(is_training=True, synthetic_size=120)
        val_ds = UrbanSoundDataset(is_training=False, synthetic_size=40)
    else:
        # Folds 1-8 for train, fold 9 for validation (fold 10 reserved for test in Phase 3)
        train_folds = [1, 2, 3, 4, 5, 6, 7, 8]
        val_folds = [9]
        train_ds = UrbanSoundDataset(folds=train_folds, is_training=args.augment)
        val_ds = UrbanSoundDataset(folds=val_folds, is_training=False)

    train_loader, val_loader = create_dataloaders(
        train_ds,
        val_ds,
        batch_size=args.batch_size,
    )

    # Initialize model
    model = UrbanSoundCNN(num_classes=settings.NUM_CLASSES)
    param_count = model.count_parameters()
    print(f"  Model:         UrbanSoundCNN (4-Block Conv2D)")
    print(f"  Parameters:    {param_count:,} trainable weights")
    print("=" * 60)

    # Run training
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        lr=args.lr,
        save_path=args.save_path,
        verbose=True,
    )

    print("-" * 60)
    best_acc = max(history["val_acc"]) if history["val_acc"] else 0.0
    print(f"[SUCCESS] Training complete! Best Validation Accuracy: {best_acc:.2f}%")
    print(f"          Model saved to: {args.save_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
