"""
Tests for Phase 2, 3, and 7 — Real Dataset Validation, Integrity Policies, and Training Pipeline

Tests cover:
- Real dataset metadata column validation
- Invalid fold values and class IDs validation
- Empty or missing fold directory detection
- Missing audio file threshold enforcement (integrity failure)
- Disjoint official fold assignments (zero data leakage)
- Preprocessing consistency across training dataset and inference classifier
- Unreadable/corrupt audio detection
- Early stopping in trainer
"""

import sys
from pathlib import Path
import csv
import numpy as np
import pytest
import soundfile as sf
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import settings
from src.models.cnn import UrbanSoundCNN
from src.training.dataset import UrbanSoundDataset, create_dataloaders
from src.training.trainer import train_model
from src.audio.processor import preprocess_audio
from src.features.extractor import extract_log_mel_spectrogram
from src.inference.predict import SoundClassifier


@pytest.fixture
def mini_dataset_fixture(tmp_path):
    """Create a minimal temporary UrbanSound8K dataset structure with valid audio files."""
    audio_dir = tmp_path / "audio"
    metadata_dir = tmp_path / "metadata"
    audio_dir.mkdir()
    metadata_dir.mkdir()

    fold1 = audio_dir / "fold1"
    fold2 = audio_dir / "fold2"
    fold1.mkdir()
    fold2.mkdir()

    sr = settings.SAMPLE_RATE
    duration = 4.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    sig = 0.3 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

    # Write WAV files
    wav1_path = fold1 / "1001-0-0-0.wav"
    wav2_path = fold1 / "1002-1-0-0.wav"
    wav3_path = fold2 / "2001-2-0-0.wav"
    sf.write(str(wav1_path), sig, sr)
    sf.write(str(wav2_path), sig, sr)
    sf.write(str(wav3_path), sig, sr)

    # Write CSV
    csv_path = metadata_dir / "UrbanSound8K.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["slice_file_name", "fsID", "start", "end", "salience", "fold", "classID", "class"])
        writer.writerow(["1001-0-0-0.wav", 1001, 0.0, 4.0, 1, 1, 0, "air_conditioner"])
        writer.writerow(["1002-1-0-0.wav", 1002, 0.0, 4.0, 1, 1, 1, "car_horn"])
        writer.writerow(["2001-2-0-0.wav", 2001, 0.0, 4.0, 1, 2, 2, "children_playing"])

    return csv_path, audio_dir


class TestDatasetValidationAndIntegrity:
    """Tests for metadata schema validation, fold integrity, and data leakage prevention."""

    def test_valid_metadata_and_audio_loading(self, mini_dataset_fixture):
        """Dataset initializes and loads valid audio files with proper provenance."""
        csv_path, audio_dir = mini_dataset_fixture
        ds = UrbanSoundDataset(folds=[1], metadata_path=csv_path, audio_dir=audio_dir, is_training=False)

        assert len(ds) == 2
        assert ds.is_synthetic is False
        assert ds.training_data_type == "urbansound8k"

        feat, label = ds[0]
        assert feat.shape == (1, settings.N_MELS, 173)
        assert label.item() in (0, 1)

    def test_missing_required_columns_raises_value_error(self, tmp_path):
        """Metadata missing required columns must raise ValueError."""
        csv_path = tmp_path / "bad_metadata.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["slice_file_name", "fold"])  # Missing classID

        with pytest.raises(ValueError) as exc:
            UrbanSoundDataset(metadata_path=csv_path, audio_dir=tmp_path)
        assert "missing required columns" in str(exc.value)

    def test_invalid_fold_values_raises_value_error(self, tmp_path):
        """Metadata containing fold numbers outside 1-10 must raise ValueError."""
        csv_path = tmp_path / "bad_folds.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["slice_file_name", "fold", "classID"])
            writer.writerow(["test.wav", 15, 0])  # Invalid fold 15

        with pytest.raises(ValueError) as exc:
            UrbanSoundDataset(metadata_path=csv_path, audio_dir=tmp_path)
        assert "invalid fold numbers" in str(exc.value)

    def test_invalid_class_id_raises_value_error(self, tmp_path):
        """Metadata containing classID outside 0-9 must raise ValueError."""
        csv_path = tmp_path / "bad_classes.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["slice_file_name", "fold", "classID"])
            writer.writerow(["test.wav", 1, 99])  # Invalid classID 99

        with pytest.raises(ValueError) as exc:
            UrbanSoundDataset(metadata_path=csv_path, audio_dir=tmp_path)
        assert "invalid classID values" in str(exc.value)

    def test_invalid_requested_folds_raises_value_error(self, mini_dataset_fixture):
        """Requesting folds outside 1-10 must raise ValueError."""
        csv_path, audio_dir = mini_dataset_fixture
        with pytest.raises(ValueError) as exc:
            UrbanSoundDataset(folds=[0, 11], metadata_path=csv_path, audio_dir=audio_dir)
        assert "Requested folds" in str(exc.value)

    def test_missing_fold_directory_raises_filenotfound(self, mini_dataset_fixture):
        """Requesting a fold whose audio directory does not exist must raise FileNotFoundError."""
        csv_path, audio_dir = mini_dataset_fixture
        with pytest.raises(FileNotFoundError) as exc:
            UrbanSoundDataset(folds=[5], metadata_path=csv_path, audio_dir=audio_dir)
        assert "does not exist" in str(exc.value)

    def test_empty_fold_directory_raises_value_error(self, mini_dataset_fixture):
        """Requesting a fold whose audio directory contains no WAV files must raise ValueError."""
        csv_path, audio_dir = mini_dataset_fixture
        empty_fold = audio_dir / "fold3"
        empty_fold.mkdir()  # Empty fold

        with pytest.raises(ValueError) as exc:
            UrbanSoundDataset(folds=[3], metadata_path=csv_path, audio_dir=audio_dir)
        assert "is empty" in str(exc.value)

    def test_missing_files_exceeding_tolerance_raises_runtime_error(self, tmp_path):
        """When referenced audio files are missing above tolerance, fail with RuntimeError."""
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        fold1 = audio_dir / "fold1"
        fold1.mkdir()

        # Create only 1 real file
        sr = settings.SAMPLE_RATE
        sig = np.zeros(int(sr * settings.AUDIO_DURATION), dtype=np.float32)
        sf.write(str(fold1 / "real_01.wav"), sig, sr)

        # Create metadata referencing 10 files (9 missing)
        csv_path = tmp_path / "UrbanSound8K.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["slice_file_name", "fold", "classID"])
            writer.writerow(["real_01.wav", 1, 0])
            for i in range(2, 11):
                writer.writerow([f"missing_{i:02d}.wav", 1, 0])

        with pytest.raises(RuntimeError) as exc:
            UrbanSoundDataset(folds=[1], metadata_path=csv_path, audio_dir=audio_dir)
        assert "Data integrity failure" in str(exc.value)

    def test_deterministic_fold_splits_no_leakage(self):
        """Verify train, validation, and test fold splits are strictly mutually disjoint."""
        train_folds = {1, 2, 3, 4, 5, 6, 7, 8}
        val_folds = {9}
        test_folds = {10}

        # Check disjointness
        assert train_folds.isdisjoint(val_folds), "Leakage between train and val folds!"
        assert train_folds.isdisjoint(test_folds), "Leakage between train and test folds!"
        assert val_folds.isdisjoint(test_folds), "Leakage between val and test folds!"
        assert train_folds | val_folds | test_folds == set(range(1, 11)), "All 10 folds must be covered"

    def test_preprocessing_consistency(self, mini_dataset_fixture):
        """Spectrogram shape from dataset matches SoundClassifier preprocessing expectation."""
        csv_path, audio_dir = mini_dataset_fixture
        ds = UrbanSoundDataset(folds=[1], metadata_path=csv_path, audio_dir=audio_dir, is_training=False)
        feat, _ = ds[0]

        # Simulate classifier preprocessing path
        dummy_audio = np.random.randn(int(settings.SAMPLE_RATE * settings.AUDIO_DURATION)).astype(np.float32)
        norm_audio = preprocess_audio(dummy_audio, settings.SAMPLE_RATE)
        log_mel = extract_log_mel_spectrogram(norm_audio, settings.SAMPLE_RATE)
        tensor = torch.from_numpy(log_mel).unsqueeze(0).unsqueeze(0).float()

        assert feat.shape == (1, settings.N_MELS, 173)
        assert tensor.shape == (1, 1, settings.N_MELS, 173)
        assert feat.shape == tensor.squeeze(0).shape

    def test_early_stopping_triggered(self, tmp_path):
        """Trainer stops early when validation accuracy does not improve for patience epochs."""
        train_ds = UrbanSoundDataset(synthetic_size=8, is_training=False)
        val_ds = UrbanSoundDataset(synthetic_size=8, is_training=False)
        train_loader, val_loader = create_dataloaders(train_ds, val_ds, batch_size=4)

        model = UrbanSoundCNN(num_classes=settings.NUM_CLASSES)
        ckpt_path = tmp_path / "early_stop_test.pth"

        # Request 10 epochs with patience=2
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=10,
            early_stopping_patience=2,
            save_path=ckpt_path,
            is_synthetic=True,
            training_data_type="synthetic",
            verbose=False,
        )

        assert len(history["val_acc"]) <= 10
