"""
Model Evaluation and Error Analysis Module — Phase 3

Provides comprehensive acoustic classification evaluation:
- Multi-class accuracy, precision, recall, and F1-score (macro and weighted)
- Per-class performance metrics and support counts
- Confusion matrix computation and high-resolution visualization
- Pairwise class confusion analysis (identifies most confused sound pairs)
- JSON and Markdown reporting for academic documentation

Usage:
    from src.training.evaluation import evaluate_model_on_dataset, plot_confusion_matrix

    metrics = evaluate_model_on_dataset(model, test_loader, device=device)
    plot_confusion_matrix(metrics["confusion_matrix"], class_names, save_path="results/confusion_matrix.png")
"""

import json
import logging
from pathlib import Path
from typing import Optional, Sequence, Union
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

from src.config.settings import settings

logger = logging.getLogger(__name__)


def compute_classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Optional[dict[int, str]] = None,
) -> dict:
    """Compute complete classification metrics.

    Args:
        y_true: Ground truth class integer indices.
        y_pred: Predicted class integer indices.
        class_names: Mapping of integer class ID to string label.

    Returns:
        Dict containing overall summary metrics, per-class metrics, and confusion matrix.
    """
    if class_names is None:
        class_names = settings.URBAN_SOUND_CLASSES

    classes = sorted(list(class_names.keys()))
    target_names = [class_names[c] for c in classes]

    # Overall metrics
    acc = accuracy_score(y_true, y_pred)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, average="weighted", zero_division=0
    )

    # Per-class breakdown
    p_per_class, r_per_class, f1_per_class, support = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, average=None, zero_division=0
    )

    per_class_metrics = {}
    for i, cid in enumerate(classes):
        cname = class_names[cid]
        per_class_metrics[cname] = {
            "class_id": int(cid),
            "precision": round(float(p_per_class[i]), 4),
            "recall": round(float(r_per_class[i]), 4),
            "f1_score": round(float(f1_per_class[i]), 4),
            "support": int(support[i]),
        }

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    return {
        "overall": {
            "accuracy": round(float(acc), 4),
            "macro_precision": round(float(macro_p), 4),
            "macro_recall": round(float(macro_r), 4),
            "macro_f1": round(float(macro_f1), 4),
            "weighted_f1": round(float(weighted_f1), 4),
            "total_samples": len(y_true),
        },
        "per_class": per_class_metrics,
        "confusion_matrix": cm.tolist(),
        "class_names": target_names,
    }


def analyze_confusions(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Optional[dict[int, str]] = None,
    top_k: int = 5,
) -> list[dict]:
    """Find the sound category pairs most frequently confused by the model.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        class_names: Mapping of class IDs to class names.
        top_k: Number of most confused pairs to return.

    Returns:
        List of dicts describing the top confused pairs.
    """
    if class_names is None:
        class_names = settings.URBAN_SOUND_CLASSES

    classes = sorted(list(class_names.keys()))
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    confusions = []
    for i, actual_id in enumerate(classes):
        row_sum = cm[i].sum()
        for j, pred_id in enumerate(classes):
            if i != j and cm[i, j] > 0:
                count = int(cm[i, j])
                pct = (count / row_sum * 100.0) if row_sum > 0 else 0.0
                confusions.append({
                    "actual_class": class_names[actual_id],
                    "predicted_class": class_names[pred_id],
                    "count": count,
                    "percentage_of_actual": round(pct, 2),
                })

    # Sort descending by count
    confusions.sort(key=lambda x: x["count"], reverse=True)
    return confusions[:top_k]


def plot_confusion_matrix(
    cm: Union[np.ndarray, list[list[int]]],
    class_names: Sequence[str],
    title: str = "Confusion Matrix",
    figsize: tuple[int, int] = (10, 8),
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> None:
    """Plot and save a high-resolution Confusion Matrix heatmap.

    Args:
        cm: Confusion matrix array of shape (num_classes, num_classes).
        class_names: List of class labels.
        title: Plot title.
        figsize: Figure dimensions (width, height).
        save_path: File path to save PNG.
        show: Whether to display interactive window.
    """
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=figsize)

    cax = ax.matshow(cm, cmap="Blues", interpolation="nearest")
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)

    num_classes = len(class_names)
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(class_names, rotation=45, ha="left", fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)

    ax.set_xlabel("Predicted Sound Class", fontsize=11, fontweight="bold", labelpad=10)
    ax.set_ylabel("True Sound Class", fontsize=11, fontweight="bold", labelpad=10)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=20)

    # Annotate numbers in cells
    thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0
    for i in range(num_classes):
        for j in range(num_classes):
            val = cm[i, j]
            color = "white" if val > thresh else "black"
            ax.text(j, i, f"{val}", ha="center", va="center", color=color, fontsize=8)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        logger.info(f"Confusion matrix plot saved: {save_path}")

    if show:
        plt.show()
    plt.close(fig)


def evaluate_model_on_dataset(
    model: nn.Module,
    dataloader: DataLoader,
    device: Optional[torch.device] = None,
    class_names: Optional[dict[int, str]] = None,
) -> dict:
    """Run evaluation pass over a DataLoader and calculate full metric report.

    Args:
        model: PyTorch model.
        dataloader: Test or validation DataLoader.
        device: Device to run evaluation on.
        class_names: Class labels dict.

    Returns:
        Dictionary with all metrics, top confusions, and raw predictions.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy().tolist())
            all_targets.extend(targets.numpy().tolist())

    metrics = compute_classification_metrics(all_targets, all_preds, class_names=class_names)
    confusions = analyze_confusions(all_targets, all_preds, class_names=class_names, top_k=5)

    metrics["top_confusions"] = confusions
    metrics["raw_predictions"] = all_preds
    metrics["raw_targets"] = all_targets
    return metrics


def export_evaluation_report(
    metrics: dict,
    json_path: Union[str, Path],
    markdown_path: Optional[Union[str, Path]] = None,
) -> None:
    """Export evaluation metrics to JSON and Markdown summary tables."""
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare JSON serializable copy
    exportable = {
        "overall": metrics["overall"],
        "per_class": metrics["per_class"],
        "top_confusions": metrics.get("top_confusions", []),
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(exportable, f, indent=2)
    logger.info(f"Evaluation JSON saved: {json_path}")

    if markdown_path:
        markdown_path = Path(markdown_path)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)

        lines = [
            "# Model Evaluation Report",
            "",
            "## Overall Performance",
            "",
            f"- **Accuracy**: {metrics['overall']['accuracy'] * 100:.2f}%",
            f"- **Macro Precision**: {metrics['overall']['macro_precision'] * 100:.2f}%",
            f"- **Macro Recall**: {metrics['overall']['macro_recall'] * 100:.2f}%",
            f"- **Macro F1-Score**: {metrics['overall']['macro_f1'] * 100:.2f}%",
            f"- **Weighted F1-Score**: {metrics['overall']['weighted_f1'] * 100:.2f}%",
            f"- **Total Test Samples**: {metrics['overall']['total_samples']}",
            "",
            "## Per-Class Metrics",
            "",
            "| Class Name | Precision | Recall | F1-Score | Support |",
            "|:-----------|:---------:|:------:|:--------:|:-------:|",
        ]

        for cname, stats in metrics["per_class"].items():
            lines.append(
                f"| `{cname}` | {stats['precision']*100:5.1f}% | "
                f"{stats['recall']*100:5.1f}% | {stats['f1_score']*100:5.1f}% | "
                f"{stats['support']:5d} |"
            )

        lines.extend([
            "",
            "## Top Confused Sound Pairs",
            "",
            "| Actual Sound | Misclassified As | Error Count | % of Actual |",
            "|:-------------|:----------------|:-----------:|:-----------:|",
        ])

        for conf in metrics.get("top_confusions", []):
            lines.append(
                f"| `{conf['actual_class']}` | `{conf['predicted_class']}` | "
                f"{conf['count']} | {conf['percentage_of_actual']:.1f}% |"
            )

        lines.append("")

        with open(markdown_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        logger.info(f"Evaluation Markdown saved: {markdown_path}")
