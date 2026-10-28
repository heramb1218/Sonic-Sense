"""
Model Evaluation CLI Script — Phase 3

Evaluates a trained UrbanSoundCNN model checkpoint on the test set
(e.g., official fold 10 or synthetic test set).

Generates:
1. Accuracy, Precision, Recall, Macro/Weighted F1 metrics
2. Per-class performance breakdown
3. Top confused acoustic categories
4. High-resolution Confusion Matrix plot saved to results/confusion_matrix.png
5. JSON report saved to results/evaluation_report.json
6. Markdown report saved to results/evaluation_report.md and appended to docs/experiments.md

Usage:
    python scripts/evaluate_model.py
    python scripts/evaluate_model.py --synthetic --test-size 80
    python scripts/evaluate_model.py --model models/cnn_urban_sound.pth --fold 10
"""

import sys
import argparse
import logging
from pathlib import Path
import torch
from torch.utils.data import DataLoader

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import settings
from src.models.cnn import UrbanSoundCNN
from src.training.dataset import UrbanSoundDataset
from src.training.evaluation import (
    evaluate_model_on_dataset,
    plot_confusion_matrix,
    export_evaluation_report,
)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def update_experiments_log(metrics: dict, model_path: Path) -> None:
    """Append structured results into docs/experiments.md."""
    exp_file = PROJECT_ROOT / "docs" / "experiments.md"
    if not exp_file.exists():
        return

    content = exp_file.read_text(encoding="utf-8")
    overall = metrics["overall"]

    entry = f"""
### Experiment Evaluation
- **Checkpoint**: `{model_path.name}`
- **Test Samples**: {overall['total_samples']}
- **Accuracy**: {overall['accuracy'] * 100:.2f}%
- **Macro Precision**: {overall['macro_precision'] * 100:.2f}%
- **Macro Recall**: {overall['macro_recall'] * 100:.2f}%
- **Macro F1-Score**: {overall['macro_f1'] * 100:.2f}%
- **Weighted F1-Score**: {overall['weighted_f1'] * 100:.2f}%
- **Artifacts**:
  - Confusion Matrix: `results/confusion_matrix.png`
  - Full Report: `results/evaluation_report.json`
"""

    if entry.strip() not in content:
        updated_content = content + "\n" + entry
        exp_file.write_text(updated_content, encoding="utf-8")
        logger.info(f"Updated experiment log: {exp_file}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained CNN sound classifier.")
    parser.add_argument("--model", type=str, default=str(settings.DEFAULT_MODEL_PATH), help="Path to .pth checkpoint")
    parser.add_argument("--fold", type=int, default=10, help="Test fold number (1-10)")
    parser.add_argument("--synthetic", action="store_true", help="Force synthetic test dataset for rapid testing")
    parser.add_argument("--test-size", type=int, default=100, help="Size of synthetic test set")
    parser.add_argument("--batch-size", type=int, default=32, help="Evaluation batch size")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Evaluation hardware")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"[ERROR] Checkpoint not found at: {model_path}")
        print("Please train a model first using `python scripts/train_cnn.py`.")
        sys.exit(1)

    print("=" * 65)
    print("ACOUSTIC CLASSIFICATION MODEL EVALUATION (Phase 2 & 3)")
    print("=" * 65)
    print(f"  Model Checkpoint: {model_path}")
    print(f"  Test Split:       {'Synthetic (' + str(args.test_size) + ' samples)' if args.synthetic else 'Fold ' + str(args.fold)}")
    print("-" * 65)

    # 1. Load Model with metadata verification
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model = UrbanSoundCNN(num_classes=settings.NUM_CLASSES)
    try:
        checkpoint = torch.load(str(model_path), map_location=device, weights_only=False)
    except Exception as e:
        print(f"[ERROR] Failed to load checkpoint: {e}")
        sys.exit(1)

    if isinstance(checkpoint, dict):
        arch = checkpoint.get("architecture")
        if arch is not None and arch != "UrbanSoundCNN":
            print(f"[ERROR] Incompatible architecture in checkpoint: '{arch}'. Expected 'UrbanSoundCNN'.")
            sys.exit(1)
        num_cls = checkpoint.get("num_classes")
        if num_cls is not None and num_cls != settings.NUM_CLASSES:
            print(f"[ERROR] Incompatible number of classes in checkpoint: {num_cls}. Expected {settings.NUM_CLASSES}.")
            sys.exit(1)
        
        # Provenance check and diagnostics
        is_synth = checkpoint.get("is_synthetic", False)
        data_type = checkpoint.get("training_data_type", "unverified")
        if is_synth or data_type == "synthetic":
            print("[WARNING] Checkpoint provenance: SYNTHETIC (smoke/development model).")
            print("          This model was NOT trained on real UrbanSound8K audio.")
        elif data_type == "urbansound8k":
            print("[INFO] Checkpoint provenance: VERIFIED UrbanSound8K trained model.")
        else:
            print("[WARNING] Checkpoint provenance: UNVERIFIED / LEGACY (missing training_data_type metadata).")

        state_dict = checkpoint.get("model_state_dict", checkpoint)
    else:
        print("[WARNING] Checkpoint provenance: RAW STATE_DICT (no training metadata).")
        state_dict = checkpoint

    try:
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"[ERROR] Failed to load model weights: {e}")
        sys.exit(1)

    model.to(device)
    model.eval()

    # 2. Prepare Test DataLoader
    if args.synthetic:
        test_ds = UrbanSoundDataset(is_training=False, synthetic_size=args.test_size)
    else:
        metadata_file = settings.URBANSOUND8K_METADATA
        audio_folder = settings.URBANSOUND8K_AUDIO_DIR
        if not metadata_file.exists() or not audio_folder.exists():
            print("[ERROR] UrbanSound8K dataset not found at expected location:")
            print(f"        {settings.URBANSOUND8K_DIR}")
            print("To evaluate on synthetic test data for verification, pass `--synthetic --test-size 100`.")
            sys.exit(1)

        try:
            test_ds = UrbanSoundDataset(folds=[args.fold], is_training=False)
        except Exception as e:
            print(f"[ERROR] Failed to load test fold {args.fold}: {e}")
            sys.exit(1)

    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # 3. Compute Metrics & Error Analysis
    print(f"[INFO] Computing predictions across {len(test_ds)} test samples on {device}...")
    metrics = evaluate_model_on_dataset(
        model=model,
        dataloader=test_loader,
        device=device,
        class_names=settings.URBAN_SOUND_CLASSES,
    )

    overall = metrics["overall"]
    print()
    print("=" * 65)
    print("GLOBAL PERFORMANCE SUMMARY")
    print("=" * 65)
    print(f"  Total Test Samples:   {overall['total_samples']}")
    print(f"  Overall Accuracy:     {overall['accuracy'] * 100:6.2f}%")
    print(f"  Macro Precision:      {overall['macro_precision'] * 100:6.2f}%")
    print(f"  Macro Recall:         {overall['macro_recall'] * 100:6.2f}%")
    print(f"  Macro F1-Score:       {overall['macro_f1'] * 100:6.2f}%")
    print(f"  Weighted F1-Score:    {overall['weighted_f1'] * 100:6.2f}%")
    print("-" * 65)

    # 4. Per-Class Breakdown
    print()
    print("PER-CLASS EVALUATION BREAKDOWN:")
    print(f"  {'Class Name':<20} {'Precision':<11} {'Recall':<11} {'F1-Score':<11} {'Support'}")
    print("  " + "-" * 60)
    for cname, stats in metrics["per_class"].items():
        print(
            f"  {cname:<20} "
            f"{stats['precision']*100:6.1f}%    "
            f"{stats['recall']*100:6.1f}%    "
            f"{stats['f1_score']*100:6.1f}%    "
            f"{stats['support']:5d}"
        )
    print("=" * 65)

    # 5. Top Confused Classes
    print()
    print("MOST FREQUENT ACOUSTIC CONFUSIONS:")
    top_conf = metrics.get("top_confusions", [])
    if top_conf:
        for idx, item in enumerate(top_conf, 1):
            print(
                f"  {idx}. Actual: {item['actual_class']:<18} "
                f"-> Predicted: {item['predicted_class']:<18} "
                f"({item['count']} times, {item['percentage_of_actual']:.1f}% of actual)"
            )
    else:
        print("  No misclassifications recorded!")
    print("=" * 65)

    # 6. Save Artifacts
    cm_path = settings.RESULTS_DIR / "confusion_matrix.png"
    json_path = settings.RESULTS_DIR / "evaluation_report.json"
    md_path = settings.RESULTS_DIR / "evaluation_report.md"

    class_names = [settings.URBAN_SOUND_CLASSES[i] for i in sorted(settings.URBAN_SOUND_CLASSES.keys())]
    plot_confusion_matrix(
        cm=metrics["confusion_matrix"],
        class_names=class_names,
        title="Urban Sound Classification — Confusion Matrix",
        save_path=cm_path,
        show=False,
    )

    export_evaluation_report(metrics, json_path=json_path, markdown_path=md_path)
    update_experiments_log(metrics, model_path)

    print()
    print("[SUCCESS] Evaluation complete!")
    print(f"  - Confusion Matrix: {cm_path}")
    print(f"  - JSON Report:      {json_path}")
    print(f"  - Markdown Report:  {md_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
