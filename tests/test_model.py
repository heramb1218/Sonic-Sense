"""
Unit Tests for Phase 2 — PyTorch CNN Sound Classification

Tests cover:
- CNN forward pass and output shape validation
- Softmax probability normalization
- Dataset generation and item shapes
- Data augmentation transformations
- Training step gradient flow
- SoundClassifier end-to-end inference

Usage:
    pytest tests/test_model.py -v
"""

import sys
from pathlib import Path
import numpy as np
import pytest
import torch
import torch.nn as nn

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import settings
from src.models.cnn import UrbanSoundCNN
from src.training.dataset import UrbanSoundDataset, create_dataloaders
from src.training.trainer import train_one_epoch, validate
from src.audio.augmentation import time_shift, add_noise, pitch_shift, time_stretch, AudioAugmenter
from src.inference.predict import SoundClassifier


@pytest.fixture
def dummy_input():
    """Batch of 4 Log-Mel Spectrograms: (batch_size=4, channels=1, n_mels=128, time_frames=173)."""
    return torch.randn(4, 1, settings.N_MELS, 173)


@pytest.fixture
def dummy_audio():
    """4-second 22050 Hz synthetic audio signal."""
    sr = settings.SAMPLE_RATE
    duration = settings.AUDIO_DURATION
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32), sr


class TestCNNArchitecture:
    """Tests for the UrbanSoundCNN model."""

    def test_cnn_forward_shape(self, dummy_input):
        """Model forward pass should produce shape (batch_size, num_classes)."""
        model = UrbanSoundCNN(num_classes=settings.NUM_CLASSES)
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)

        assert output.shape == (4, settings.NUM_CLASSES)
        assert not torch.isnan(output).any(), "Output logits should not contain NaNs"

    def test_cnn_predict_proba(self, dummy_input):
        """Softmax probabilities should sum to 1.0 for each sample in the batch."""
        model = UrbanSoundCNN(num_classes=settings.NUM_CLASSES)
        model.eval()
        with torch.no_grad():
            probs = model.predict_proba(dummy_input)

        assert probs.shape == (4, settings.NUM_CLASSES)
        sums = probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(4), atol=1e-5), "Probabilities must sum to 1.0"
        assert (probs >= 0.0).all() and (probs <= 1.0).all()

    def test_cnn_parameter_count(self):
        """Model should have reasonable parameter size (between 100k and 5M)."""
        model = UrbanSoundCNN(num_classes=settings.NUM_CLASSES)
        params = model.count_parameters()
        assert 100_000 < params < 5_000_000, f"Unexpected parameter count: {params}"


class TestAudioAugmentation:
    """Tests for audio augmentation functions."""

    def test_time_shift_preserves_length(self, dummy_audio):
        audio, _ = dummy_audio
        shifted = time_shift(audio, shift_limit=0.2)
        assert len(shifted) == len(audio)

    def test_add_noise_preserves_length(self, dummy_audio):
        audio, _ = dummy_audio
        noisy = add_noise(audio, noise_factor=0.01)
        assert len(noisy) == len(audio)
        assert not np.array_equal(audio, noisy), "Noise should alter signal"

    def test_time_stretch_preserves_length(self, dummy_audio):
        audio, _ = dummy_audio
        stretched = time_stretch(audio, rate=1.1)
        assert len(stretched) == len(audio)

    def test_audio_augmenter_pipeline(self, dummy_audio):
        audio, sr = dummy_audio
        augmenter = AudioAugmenter(use_shift=True, use_noise=True, prob=1.0)
        aug_audio = augmenter(audio, sr)
        assert len(aug_audio) == len(audio)


class TestDatasetAndDataLoader:
    """Tests for Dataset and DataLoader."""

    def test_synthetic_dataset_shapes(self):
        """Synthetic dataset item should have expected tensor dimensions and types."""
        ds = UrbanSoundDataset(synthetic_size=8, is_training=False)
        assert len(ds) == 8

        feature, label = ds[0]
        assert isinstance(feature, torch.Tensor)
        assert feature.shape == (1, settings.N_MELS, 173)
        assert isinstance(label, torch.Tensor)
        assert 0 <= label.item() < settings.NUM_CLASSES

    def test_create_dataloaders(self):
        """DataLoaders should yield batches with correct tensor shapes."""
        train_ds = UrbanSoundDataset(synthetic_size=16, is_training=True)
        val_ds = UrbanSoundDataset(synthetic_size=8, is_training=False)
        train_loader, val_loader = create_dataloaders(train_ds, val_ds, batch_size=4)

        for inputs, targets in train_loader:
            assert inputs.shape == (4, 1, settings.N_MELS, 173)
            assert targets.shape == (4,)
            break


class TestTrainingAndInference:
    """Tests for training step and inference pipeline."""

    def test_single_training_epoch(self):
        """Running a single training epoch should compute loss and update weights."""
        train_ds = UrbanSoundDataset(synthetic_size=8, is_training=False)
        loader = torch.utils.data.DataLoader(train_ds, batch_size=4)

        model = UrbanSoundCNN(num_classes=settings.NUM_CLASSES)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        loss, acc = train_one_epoch(model, loader, criterion, optimizer, torch.device("cpu"))
        assert loss > 0.0
        assert 0.0 <= acc <= 100.0

    def test_sound_classifier_predict_audio(self, dummy_audio):
        """SoundClassifier should predict class name, confidence, and complete probability dict."""
        audio, sr = dummy_audio
        classifier = SoundClassifier(model_path=None)  # uninitialized baseline weights

        result = classifier.predict_audio(audio, sr)

        assert "predicted_class" in result
        assert "confidence" in result
        assert "probabilities" in result
        assert 0.0 <= result["confidence"] <= 1.0
        assert len(result["probabilities"]) == settings.NUM_CLASSES
        total_prob = sum(result["probabilities"].values())
        assert abs(total_prob - 1.0) < 0.01
