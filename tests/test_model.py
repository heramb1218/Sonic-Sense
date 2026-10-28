"""
Unit Tests for Phase 2 — PyTorch CNN Sound Classification and Integration

Tests cover:
- CNN forward pass and output shape validation
- Softmax probability normalization
- Model parameter count bounds
- Dataset generation and item shapes
- Data augmentation transformations
- Class weight calculations for imbalanced datasets
- Explicit error on missing real dataset (no silent synthetic fallback)
- Training step gradient flow and epoch execution
- SoundClassifier missing checkpoint error (no silent random fallback)
- SoundClassifier explicit allow_untrained opt-in
- SoundClassifier loading valid trained checkpoints and metadata
- SoundClassifier rejecting incompatible checkpoints
- SoundClassifier class mapping and probability distribution

Usage:
    pytest tests/test_model.py -v
"""

import sys
from pathlib import Path
import tempfile
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
from src.training.trainer import train_one_epoch, validate, calculate_class_weights
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


@pytest.fixture
def valid_checkpoint_file(tmp_path):
    """Create a temporary valid model checkpoint file with full metadata."""
    model = UrbanSoundCNN(num_classes=settings.NUM_CLASSES)
    ckpt_path = tmp_path / "test_cnn_model.pth"
    checkpoint = {
        "architecture": "UrbanSoundCNN",
        "num_classes": settings.NUM_CLASSES,
        "classes": settings.URBAN_SOUND_CLASSES,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": {},
        "best_val_acc": 85.5,
        "val_loss": 0.45,
        "is_trained": True,
        "is_synthetic": False,
        "training_data_type": "urbansound8k",
        "input_shape": [1, settings.N_MELS, 173],
    }
    torch.save(checkpoint, str(ckpt_path))
    return ckpt_path


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

    def test_missing_real_dataset_raises_filenotfound(self):
        """Attempting to load real dataset when files are missing must raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError) as exc_info:
            UrbanSoundDataset(
                metadata_path="non_existent_metadata.csv",
                audio_dir="non_existent_audio_dir",
            )
        assert "UrbanSound8K" in str(exc_info.value)

    def test_calculate_class_weights(self):
        """Calculate class weights returns valid positive normalized weights."""
        ds = UrbanSoundDataset(synthetic_size=30, is_training=False)
        weights = calculate_class_weights(ds, num_classes=settings.NUM_CLASSES)
        assert len(weights) == settings.NUM_CLASSES
        assert (weights > 0).all()
        assert abs(weights.mean().item() - 1.0) < 0.05


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


class TestModelIntegration:
    """Focused integration tests for model loading, validation, and error handling."""

    def test_missing_checkpoint_raises_filenotfound_by_default(self):
        """Normal inference without checkpoint must raise FileNotFoundError (no silent random fallback)."""
        fake_path = Path("models/definitely_missing_checkpoint_123.pth")
        with pytest.raises(FileNotFoundError) as exc_info:
            SoundClassifier(model_path=fake_path, allow_untrained=False)
        assert "Model checkpoint not found" in str(exc_info.value)

    def test_allow_untrained_opt_in(self, dummy_audio):
        """Explicitly passing allow_untrained=True initializes baseline model with is_trained=False."""
        fake_path = Path("models/definitely_missing_checkpoint_456.pth")
        classifier = SoundClassifier(model_path=fake_path, allow_untrained=True)
        assert classifier.is_trained is False

        audio, sr = dummy_audio
        res = classifier.predict_audio(audio, sr)
        assert res["is_trained"] is False
        assert res["predicted_class"] in settings.URBAN_SOUND_CLASSES.values()

    def test_load_valid_checkpoint(self, valid_checkpoint_file, dummy_audio):
        """SoundClassifier loads valid checkpoint, metadata, and marks is_trained=True."""
        classifier = SoundClassifier(model_path=valid_checkpoint_file)
        assert classifier.is_trained is True

        audio, sr = dummy_audio
        result = classifier.predict_audio(audio, sr)
        assert result["is_trained"] is True
        assert 0 <= result["class_id"] < settings.NUM_CLASSES
        assert result["predicted_class"] == settings.URBAN_SOUND_CLASSES[result["class_id"]]
        assert 0.0 <= result["confidence"] <= 1.0

    def test_reject_incompatible_architecture(self, tmp_path):
        """Checkpoint with mismatched architecture string must raise ValueError."""
        bad_path = tmp_path / "bad_arch.pth"
        torch.save({"architecture": "ResNet50", "model_state_dict": {}}, str(bad_path))

        with pytest.raises(ValueError) as exc_info:
            SoundClassifier(model_path=bad_path)
        assert "Incompatible architecture" in str(exc_info.value)

    def test_reject_incompatible_num_classes(self, tmp_path):
        """Checkpoint with mismatched num_classes must raise ValueError."""
        bad_path = tmp_path / "bad_num_classes.pth"
        torch.save({"num_classes": 5, "model_state_dict": {}}, str(bad_path))

        with pytest.raises(ValueError) as exc_info:
            SoundClassifier(model_path=bad_path)
        assert "Incompatible num_classes" in str(exc_info.value)

    def test_class_mapping_and_probabilities(self, valid_checkpoint_file, dummy_audio):
        """Class probabilities dictionary must contain all 10 UrbanSound8K classes summing to 1.0."""
        classifier = SoundClassifier(model_path=valid_checkpoint_file)
        audio, sr = dummy_audio
        result = classifier.predict_audio(audio, sr)

        assert len(result["probabilities"]) == settings.NUM_CLASSES
        for class_name in settings.URBAN_SOUND_CLASSES.values():
            assert class_name in result["probabilities"]
            assert 0.0 <= result["probabilities"][class_name] <= 1.0

        total = sum(result["probabilities"].values())
        assert abs(total - 1.0) < 1e-4

    def test_predict_file_with_sample_wav(self, valid_checkpoint_file):
        """predict_file must decode sample .wav and return consistent result dictionary."""
        sample_path = PROJECT_ROOT / "data" / "sample" / "sample_siren_urban.wav"
        if sample_path.exists():
            classifier = SoundClassifier(model_path=valid_checkpoint_file)
            result = classifier.predict_file(sample_path)
            assert "predicted_class" in result
            assert "confidence" in result
            assert "file_path" in result
            assert result["is_trained"] is True


class TestCheckpointProvenanceAndOverwriteSafety:
    """Unit and integration tests for Phase 2.1: Checkpoint Provenance and Overwrite Safety."""

    def test_synthetic_dataset_provenance(self):
        """UrbanSoundDataset with synthetic_size > 0 must explicitly declare synthetic provenance."""
        ds = UrbanSoundDataset(synthetic_size=5, is_training=False)
        assert ds.is_synthetic is True
        assert ds.training_data_type == "synthetic"

    def test_trainer_synthetic_metadata_and_not_trained(self, tmp_path):
        """train_model on synthetic data must save is_synthetic=True and is_trained=False."""
        from src.training.trainer import train_model

        ckpt_path = tmp_path / "synthetic_test.pth"
        train_ds = UrbanSoundDataset(synthetic_size=4, is_training=True)
        val_ds = UrbanSoundDataset(synthetic_size=4, is_training=False)
        train_loader, val_loader = create_dataloaders(train_ds, val_ds, batch_size=2)
        model = UrbanSoundCNN(num_classes=settings.NUM_CLASSES)

        train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=1,
            save_path=ckpt_path,
            is_synthetic=True,
            training_data_type="synthetic",
            verbose=False,
        )

        assert ckpt_path.exists()
        saved = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        assert saved["is_synthetic"] is True
        assert saved["training_data_type"] == "synthetic"
        assert saved["is_trained"] is False
        assert saved["best_val_acc"] is not None

    def test_trainer_refuses_overwrite_without_flag(self, tmp_path):
        """train_model must raise FileExistsError if checkpoint exists and overwrite=False."""
        from src.training.trainer import train_model

        ckpt_path = tmp_path / "existing.pth"
        ckpt_path.write_text("existing content", encoding="utf-8")

        train_ds = UrbanSoundDataset(synthetic_size=4, is_training=True)
        val_ds = UrbanSoundDataset(synthetic_size=4, is_training=False)
        train_loader, val_loader = create_dataloaders(train_ds, val_ds, batch_size=2)
        model = UrbanSoundCNN(num_classes=settings.NUM_CLASSES)

        with pytest.raises(FileExistsError) as exc_info:
            train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=1,
                save_path=ckpt_path,
                overwrite=False,
                verbose=False,
            )
        assert "already exists" in str(exc_info.value)
        assert "overwrite=True" in str(exc_info.value)

    def test_trainer_permits_overwrite_when_explicit(self, tmp_path):
        """train_model succeeds in overwriting an existing file when overwrite=True is passed."""
        from src.training.trainer import train_model

        ckpt_path = tmp_path / "existing.pth"
        ckpt_path.write_text("temporary content", encoding="utf-8")

        train_ds = UrbanSoundDataset(synthetic_size=4, is_training=True)
        val_ds = UrbanSoundDataset(synthetic_size=4, is_training=False)
        train_loader, val_loader = create_dataloaders(train_ds, val_ds, batch_size=2)
        model = UrbanSoundCNN(num_classes=settings.NUM_CLASSES)

        train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=1,
            save_path=ckpt_path,
            overwrite=True,
            is_synthetic=True,
            training_data_type="synthetic",
            verbose=False,
        )
        assert ckpt_path.exists()
        saved = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        assert saved["is_synthetic"] is True

    def test_normal_inference_rejects_synthetic_checkpoint(self, tmp_path):
        """Normal inference (allow_untrained=False) must raise ValueError for synthetic checkpoints."""
        model = UrbanSoundCNN(num_classes=settings.NUM_CLASSES)
        ckpt_path = tmp_path / "synthetic_model.pth"
        torch.save(
            {
                "architecture": "UrbanSoundCNN",
                "num_classes": settings.NUM_CLASSES,
                "classes": settings.URBAN_SOUND_CLASSES,
                "model_state_dict": model.state_dict(),
                "is_trained": False,
                "is_synthetic": True,
                "training_data_type": "synthetic",
            },
            str(ckpt_path),
        )

        with pytest.raises(ValueError) as exc_info:
            SoundClassifier(model_path=ckpt_path, allow_untrained=False)
        assert "trained on synthetic data" in str(exc_info.value)
        assert "cannot be used for genuine acoustic classification" in str(exc_info.value)

    def test_normal_inference_rejects_legacy_unverified_checkpoint(self, tmp_path):
        """Normal inference must reject checkpoints lacking verified UrbanSound8K provenance."""
        model = UrbanSoundCNN(num_classes=settings.NUM_CLASSES)
        ckpt_path = tmp_path / "unverified_model.pth"
        torch.save(
            {
                "architecture": "UrbanSoundCNN",
                "num_classes": settings.NUM_CLASSES,
                "classes": settings.URBAN_SOUND_CLASSES,
                "model_state_dict": model.state_dict(),
                "is_trained": True,  # Old format claiming trained without provenance
            },
            str(ckpt_path),
        )

        with pytest.raises(ValueError) as exc_info:
            SoundClassifier(model_path=ckpt_path, allow_untrained=False)
        assert "lacks verified UrbanSound8K training data provenance" in str(exc_info.value)

    def test_development_mode_allows_synthetic_checkpoint(self, tmp_path, dummy_audio):
        """Development mode (allow_untrained=True) permits synthetic checkpoints with is_trained=False."""
        model = UrbanSoundCNN(num_classes=settings.NUM_CLASSES)
        ckpt_path = tmp_path / "synthetic_model.pth"
        torch.save(
            {
                "architecture": "UrbanSoundCNN",
                "num_classes": settings.NUM_CLASSES,
                "classes": settings.URBAN_SOUND_CLASSES,
                "model_state_dict": model.state_dict(),
                "is_trained": False,
                "is_synthetic": True,
                "training_data_type": "synthetic",
            },
            str(ckpt_path),
        )

        classifier = SoundClassifier(model_path=ckpt_path, allow_untrained=True)
        assert classifier.is_trained is False
        assert classifier.is_synthetic is True
        assert classifier.training_data_type == "synthetic"

        audio, sr = dummy_audio
        pred = classifier.predict_audio(audio, sr)
        assert pred["is_trained"] is False
        assert pred["is_synthetic"] is True
        assert pred["training_data_type"] == "synthetic"
        assert pred["predicted_class"] in settings.URBAN_SOUND_CLASSES.values()

    def test_inference_provenance_fields_in_output(self, valid_checkpoint_file, dummy_audio):
        """predict_audio output dictionary must contain all provenance fields."""
        classifier = SoundClassifier(model_path=valid_checkpoint_file)
        audio, sr = dummy_audio
        pred = classifier.predict_audio(audio, sr)

        assert pred["is_trained"] is True
        assert pred["is_synthetic"] is False
        assert pred["training_data_type"] == "urbansound8k"
        assert "predicted_class" in pred
        assert "confidence" in pred
        assert "probabilities" in pred

    def test_cli_blocks_synthetic_overwriting_production(self):
        """CLI must refuse synthetic training targeting models/cnn_urban_sound.pth without override flag."""
        import subprocess

        prod_path = str(settings.DEFAULT_MODEL_PATH)
        cmd = [
            sys.executable,
            "scripts/train_cnn.py",
            "--synthetic",
            "--save-path",
            prod_path,
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode != 0
        assert "Refusing to save synthetic model checkpoint to protected production path" in (res.stdout + res.stderr)

    def test_cli_blocks_overwrite_without_flag(self, tmp_path):
        """CLI must exit with error if destination file exists and --overwrite is omitted."""
        import subprocess

        existing_file = tmp_path / "exist.pth"
        existing_file.write_text("placeholder", encoding="utf-8")

        cmd = [
            sys.executable,
            "scripts/train_cnn.py",
            "--synthetic",
            "--save-path",
            str(existing_file),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode != 0
        assert "already exists" in (res.stdout + res.stderr)
