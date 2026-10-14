"""
Unit Tests for Phase 3 — Model Evaluation and Metrics

Tests verify:
- Accurate calculation of accuracy, precision, recall, and F1-score
- Confusion matrix generation and dimension verification
- Error analysis and top-confusion pair detection
- Plot generation of the confusion matrix heatmap
- JSON and Markdown report export

Usage:
    pytest tests/test_evaluation.py -v
"""

import sys
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import settings
from src.models.cnn import UrbanSoundCNN
from src.training.dataset import UrbanSoundDataset
from src.training.evaluation import (
    compute_classification_metrics,
    analyze_confusions,
    plot_confusion_matrix,
    evaluate_model_on_dataset,
    export_evaluation_report,
)


@pytest.fixture
def sample_predictions():
    """Deterministic true vs predicted label sequence with known confusion."""
    # 10 samples: 8 correct, 2 confused (class 1 predicted as 2, class 3 predicted as 4)
    y_true = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
    y_pred = [0, 2, 2, 4, 4, 0, 1, 2, 3, 4]
    return y_true, y_pred


class TestEvaluationMetrics:
    """Tests for classification metric calculations."""

    def test_compute_metrics_accuracy(self, sample_predictions):
        y_true, y_pred = sample_predictions
        metrics = compute_classification_metrics(y_true, y_pred)

        assert "overall" in metrics
        assert "per_class" in metrics
        assert "confusion_matrix" in metrics

        # 8 out of 10 correct -> accuracy 0.8
        assert metrics["overall"]["accuracy"] == 0.8
        assert 0.0 <= metrics["overall"]["macro_f1"] <= 1.0

    def test_confusion_matrix_shape(self, sample_predictions):
        y_true, y_pred = sample_predictions
        metrics = compute_classification_metrics(y_true, y_pred)
        cm = np.array(metrics["confusion_matrix"])

        assert cm.shape == (settings.NUM_CLASSES, settings.NUM_CLASSES)
        assert cm.sum() == len(y_true)

    def test_analyze_confusions(self, sample_predictions):
        y_true, y_pred = sample_predictions
        confusions = analyze_confusions(y_true, y_pred, top_k=5)

        assert len(confusions) > 0
        top = confusions[0]
        assert "actual_class" in top
        assert "predicted_class" in top
        assert "count" in top
        assert top["count"] >= 1


class TestConfusionMatrixPlotting:
    """Tests for visualization generation."""

    def test_plot_confusion_matrix_file_creation(self, tmp_path):
        cm = np.eye(settings.NUM_CLASSES, dtype=int)
        class_names = [settings.URBAN_SOUND_CLASSES[i] for i in range(settings.NUM_CLASSES)]
        out_file = tmp_path / "test_cm.png"

        plot_confusion_matrix(cm, class_names, save_path=out_file, show=False)

        assert out_file.exists()
        assert out_file.stat().st_size > 1000


class TestModelDatasetEvaluation:
    """Tests for end-to-end evaluation pipeline on a model."""

    def test_evaluate_model_on_dataset(self):
        model = UrbanSoundCNN(num_classes=settings.NUM_CLASSES)
        dataset = UrbanSoundDataset(synthetic_size=20, is_training=False)
        loader = DataLoader(dataset, batch_size=8, shuffle=False)

        metrics = evaluate_model_on_dataset(
            model=model,
            dataloader=loader,
            device=torch.device("cpu"),
        )

        assert metrics["overall"]["total_samples"] == 20
        assert 0.0 <= metrics["overall"]["accuracy"] <= 1.0
        assert len(metrics["raw_predictions"]) == 20
        assert len(metrics["raw_targets"]) == 20

    def test_export_evaluation_report(self, sample_predictions, tmp_path):
        y_true, y_pred = sample_predictions
        metrics = compute_classification_metrics(y_true, y_pred)
        metrics["top_confusions"] = analyze_confusions(y_true, y_pred)

        json_path = tmp_path / "report.json"
        md_path = tmp_path / "report.md"

        export_evaluation_report(metrics, json_path=json_path, markdown_path=md_path)

        assert json_path.exists()
        assert md_path.exists()

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert "overall" in data
            assert data["overall"]["accuracy"] == 0.8
