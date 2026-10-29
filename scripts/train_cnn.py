"""
CNN Training Script — Phase 2 & Phase 2.1 Checkpoint Safety

Trains the UrbanSoundCNN audio classifier on UrbanSound8K 10-fold dataset
with reproducible seeds, class weighting, data augmentation, provenance metadata,
and strict overwrite protection.

Default Save Paths:
    - Real-data UrbanSound8K training: models/cnn_urban_sound.pth
    - Synthetic smoke-test training:   models/dev/cnn_synthetic_smoke.pth

Usage:
    # Quick synthetic smoke test (saved to models/dev/cnn_synthetic_smoke.pth)
    python scripts/train_cnn.py --synthetic --epochs 3

    # Standard training on UrbanSound8K (saved to models/cnn_urban_sound.pth)
    python scripts/train_cnn.py --epochs 20 --augment --weighted-loss

    # Overwrite an existing checkpoint explicitly
    python scripts/train_cnn.py --synthetic --epochs 3 --overwrite
"""

import sys
import random
import argparse
import logging
from pathlib import Path
import numpy as np
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import settings
from src.models.cnn import UrbanSoundCNN
from src.training.dataset import UrbanSoundDataset, create_dataloaders
from src.training.trainer import train_model, calculate_class_weights

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_SYNTHETIC_PATH = PROJECT_ROOT / "models" / "dev" / "cnn_synthetic_smoke.pth"
DEFAULT_PRODUCTION_PATH = settings.DEFAULT_MODEL_PATH


def set_seed(seed: int = 42) -> None:
    """Set random seeds across random, numpy, and torch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description="Train PyTorch CNN for Urban Sound Classification.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Initial learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="L2 regularization weight decay")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Training hardware")
    parser.add_argument("--workers", type=int, default=0, help="DataLoader worker processes")
    parser.add_argument("--augment", action="store_true", help="Enable audio data augmentation during training")
    parser.add_argument("--weighted-loss", action="store_true", help="Use class-weighted loss to handle dataset imbalance")
    parser.add_argument("--synthetic", action="store_true", help="Force synthetic dataset mode for fast smoke test")
    parser.add_argument("--synthetic-size", type=int, default=120, help="Number of samples in synthetic training set")
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Early stopping patience (number of epochs without improvement before stopping)",
    )
    parser.add_argument(
        "--candidate",
        action="store_true",
        help="Save model as candidate checkpoint models/candidate_cnn_urban_sound.pth for evaluation before production promotion",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Destination model path (defaults to models/cnn_urban_sound.pth for real training, or models/dev/cnn_synthetic_smoke.pth for synthetic)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly permit overwriting an existing checkpoint file",
    )
    parser.add_argument(
        "--allow-synthetic-override-production",
        action="store_true",
        help="Deliberately allow a synthetic training run to target the protected production checkpoint path",
    )
    args = parser.parse_args()

    # Reproducibility
    set_seed(args.seed)

    # Hardware selection
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Resolve output checkpoint path (Task 1 & Candidate handling)
    candidate_path = PROJECT_ROOT / "models" / "candidate_cnn_urban_sound.pth"
    if args.save_path is None:
        if args.synthetic:
            save_path = DEFAULT_SYNTHETIC_PATH
        elif args.candidate:
            save_path = candidate_path
        else:
            save_path = DEFAULT_PRODUCTION_PATH
    else:
        save_path = Path(args.save_path)
        # Protect production checkpoint from accidental synthetic overwrites
        if args.synthetic:
            try:
                target_resolved = save_path.resolve()
                prod_resolved = DEFAULT_PRODUCTION_PATH.resolve()
            except Exception:
                target_resolved = save_path
                prod_resolved = DEFAULT_PRODUCTION_PATH

            if target_resolved == prod_resolved and not args.allow_synthetic_override_production:
                print("[ERROR] Refusing to save synthetic model checkpoint to protected production path:")
                print(f"        {DEFAULT_PRODUCTION_PATH}")
                print("Synthetic training runs must not overwrite the production model artifact.")
                print("To save to a custom development path, use: --save-path models/dev/my_test.pth")
                print("To deliberately force targeting production, pass: --allow-synthetic-override-production --overwrite")
                sys.exit(1)

    # Overwrite safety check (Task 1)
    if save_path.exists() and not args.overwrite:
        print(f"[ERROR] Destination checkpoint already exists: {save_path}")
        print("Refusing to overwrite existing checkpoint without explicit permission.")
        print("To overwrite this existing checkpoint, pass the --overwrite flag.")
        sys.exit(1)

    print("=" * 65)
    print("URBAN SOUND CNN TRAINING PIPELINE (Phase 2 & 2.1)")
    print("=" * 65)
    print(f"  Target Device:  {device}")
    print(f"  Random Seed:    {args.seed}")
    print(f"  Epochs:         {args.epochs}")
    print(f"  Batch Size:     {args.batch_size}")
    print(f"  Learning Rate:  {args.lr}")
    print(f"  Weight Decay:   {args.weight_decay}")
    print(f"  Augmentation:   {args.augment}")
    print(f"  Weighted Loss:  {args.weighted_loss}")
    print(f"  Synthetic Mode: {args.synthetic}")
    print(f"  Save Path:      {save_path}")
    print(f"  Allow Overwrite:{args.overwrite}")
    print("-" * 65)

    # Prepare datasets (Task 2)
    if args.synthetic:
        print("[INFO] Preparing synthetic dataset for rapid smoke testing...")
        train_ds = UrbanSoundDataset(is_training=args.augment, synthetic_size=args.synthetic_size)
        val_ds = UrbanSoundDataset(is_training=False, synthetic_size=max(20, args.synthetic_size // 3))
    else:
        # Check dataset readiness before proceeding
        metadata_file = settings.URBANSOUND8K_METADATA
        audio_folder = settings.URBANSOUND8K_AUDIO_DIR

        if not metadata_file.exists() or not audio_folder.exists():
            print("[ERROR] UrbanSound8K dataset not found!")
            print(f"        Expected metadata: {metadata_file}")
            print(f"        Expected audio:    {audio_folder}")
            print("")
            print("Expected directory structure:")
            print("  data/raw/UrbanSound8K/")
            print("  |-- audio/")
            print("  |   |-- fold1/")
            print("  |   |-- fold2/")
            print("  |   \\-- ... fold10/")
            print("  \\-- metadata/")
            print("      \\-- UrbanSound8K.csv")
            print("")
            print("To acquire and place the dataset:")
            print("  1. Download UrbanSound8K from: https://urbansounddataset.weebly.com/urbansound8k.html")
            print("  2. Extract the archive so that 'data/raw/UrbanSound8K' contains 'audio' and 'metadata'")
            print("  3. Run training: python scripts/train_cnn.py --epochs 20 --augment")
            print("")
            print("For an automated local test without the 6 GB dataset, run:")
            print("  python scripts/train_cnn.py --synthetic --epochs 3")
            print("=" * 65)
            sys.exit(1)

        train_folds = [1, 2, 3, 4, 5, 6, 7, 8]
        val_folds = [9]
        print(f"[INFO] Loading UrbanSound8K (Train Folds: {train_folds}, Val Fold: {val_folds})...")

        try:
            train_ds = UrbanSoundDataset(folds=train_folds, is_training=args.augment)
            val_ds = UrbanSoundDataset(folds=val_folds, is_training=False)
        except Exception as e:
            print(f"[ERROR] Failed to load dataset: {e}")
            sys.exit(1)

    print(f"  Train Samples:  {len(train_ds)} (Data: {train_ds.training_data_type})")
    print(f"  Val Samples:    {len(val_ds)} (Data: {val_ds.training_data_type})")

    # Class weighting if requested
    class_weights = None
    if args.weighted_loss:
        class_weights = calculate_class_weights(train_ds, num_classes=settings.NUM_CLASSES)
        print(f"  Class Weights:  {[round(w, 2) for w in class_weights.tolist()]}")

    train_loader, val_loader = create_dataloaders(
        train_ds,
        val_ds,
        batch_size=args.batch_size,
        num_workers=args.workers,
    )

    # Initialize model
    model = UrbanSoundCNN(num_classes=settings.NUM_CLASSES)
    param_count = model.count_parameters()
    print(f"  Model:          UrbanSoundCNN (4-Block Conv2D)")
    print(f"  Parameters:     {param_count:,} trainable weights")
    print("=" * 65)

    # Run training with explicit provenance and overwrite enforcement (Task 1 & 2)
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        class_weights=class_weights,
        early_stopping_patience=args.patience,
        training_data_type=train_ds.training_data_type,
        is_synthetic=train_ds.is_synthetic,
        device=device,
        save_path=save_path,
        overwrite=args.overwrite or not save_path.exists(),
        verbose=True,
    )

    print("-" * 65)
    best_acc = max(history["val_acc"]) if history["val_acc"] else 0.0
    print(f"[SUCCESS] Training complete! Best Validation Accuracy: {best_acc:.2f}%")
    print(f"          Model checkpoint saved to: {save_path}")
    print(f"          Training Data Provenance:  {train_ds.training_data_type} (is_synthetic={train_ds.is_synthetic})")
    print("=" * 65)


if __name__ == "__main__":
    main()
