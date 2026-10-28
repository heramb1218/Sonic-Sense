"""
PyTorch Dataset and DataLoader for Urban Sound Classification

Supports:
- Official UrbanSound8K 10-fold cross-validation scheme
- Loading audio, preprocessing, and extracting Log-Mel Spectrograms
- On-the-fly audio data augmentation
- Explicit opt-in synthetic mode for local testing without downloading the full dataset
- Explicit training-data provenance tracking (is_synthetic, training_data_type)

Usage:
    from src.training.dataset import UrbanSoundDataset, create_dataloaders

    train_ds = UrbanSoundDataset(folds=[1, 2, 3, 4, 5, 6, 7, 8], is_training=True)
    val_ds = UrbanSoundDataset(folds=[9], is_training=False)
    train_loader, val_loader = create_dataloaders(train_ds, val_ds, batch_size=32)
"""

import logging
from pathlib import Path
from typing import Optional, Sequence
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from src.config.settings import settings
from src.audio.processor import load_audio, preprocess_audio
from src.audio.augmentation import AudioAugmenter
from src.features.extractor import extract_log_mel_spectrogram

logger = logging.getLogger(__name__)


class UrbanSoundDataset(Dataset):
    """PyTorch Dataset for UrbanSound8K audio classification."""

    def __init__(
        self,
        folds: Optional[Sequence[int]] = None,
        metadata_path: Optional[str | Path] = None,
        audio_dir: Optional[str | Path] = None,
        is_training: bool = False,
        augmenter: Optional[AudioAugmenter] = None,
        synthetic_size: Optional[int] = None,
    ):
        """Initialize the UrbanSound8K dataset.

        Args:
            folds: List of fold numbers (1-10) to include. If None, includes all folds.
            metadata_path: Path to UrbanSound8K.csv. Defaults to settings.URBANSOUND8K_METADATA.
            audio_dir: Path to audio fold root. Defaults to settings.URBANSOUND8K_AUDIO_DIR.
            is_training: If True, enable augmentation by default.
            augmenter: Optional AudioAugmenter instance.
            synthetic_size: If specified, generates a synthetic dataset of this size
                (useful for testing and development without the full dataset).
        """
        self.is_training = is_training
        self.augmenter = augmenter if augmenter is not None else (
            AudioAugmenter(use_shift=True, use_noise=True) if is_training else None
        )
        self.synthetic_size = synthetic_size
        self.skipped_files: list[str] = []

        if synthetic_size is not None and synthetic_size > 0:
            self.is_synthetic: bool = True
            self.training_data_type: str = "synthetic"
            self.samples = self._generate_synthetic_manifest(synthetic_size)
            logger.info(f"Initialized synthetic dataset with {synthetic_size} samples.")
            return

        # Attempt to load real dataset
        metadata_path = Path(metadata_path or settings.URBANSOUND8K_METADATA)
        audio_dir = Path(audio_dir or settings.URBANSOUND8K_AUDIO_DIR)

        if not metadata_path.exists():
            raise FileNotFoundError(
                f"UrbanSound8K metadata not found at: {metadata_path}\n"
                f"Please place the dataset in '{settings.URBANSOUND8K_DIR}' with structure:\n"
                f"  {settings.URBANSOUND8K_DIR}/\n"
                f"  |-- audio/fold1/ ... fold10/\n"
                f"  \\-- metadata/UrbanSound8K.csv\n"
                f"For testing without the real dataset, pass synthetic_size=<N>."
            )

        if not audio_dir.exists():
            raise FileNotFoundError(
                f"UrbanSound8K audio directory not found at: {audio_dir}\n"
                f"Expected directory: {settings.URBANSOUND8K_AUDIO_DIR}"
            )

        df = pd.read_csv(metadata_path)
        if folds is not None:
            df = df[df["fold"].isin(folds)].copy()

        # Build records: list of (file_path, class_id)
        records = []
        for _, row in df.iterrows():
            file_path = audio_dir / f"fold{row['fold']}" / row["slice_file_name"]
            if file_path.exists():
                records.append({
                    "file_path": str(file_path),
                    "class_id": int(row["classID"]),
                    "class_name": str(row["class"]),
                })
            else:
                self.skipped_files.append(str(file_path))

        if self.skipped_files:
            logger.warning(
                f"Encountered {len(self.skipped_files)} missing audio files referenced in metadata."
            )

        if not records:
            raise FileNotFoundError(
                f"No audio files found for folds {folds} in {audio_dir}. "
                f"Check that audio fold directories contain .wav files."
            )

        # Real dataset successfully verified
        self.is_synthetic = False
        self.training_data_type = "urbansound8k"
        self.samples = records
        logger.info(f"Loaded {len(self.samples)} verified samples from folds {folds}.")

    def _generate_synthetic_manifest(self, size: int) -> list[dict]:
        """Generate manifest entries for synthetic samples."""
        records = []
        for i in range(size):
            class_id = i % settings.NUM_CLASSES
            class_name = settings.URBAN_SOUND_CLASSES.get(class_id, "unknown")
            records.append({
                "synthetic": True,
                "class_id": class_id,
                "class_name": class_name,
                "seed": i,
            })
        return records

    def _get_synthetic_audio(self, seed: int, class_id: int) -> tuple[np.ndarray, int]:
        """Generate deterministic synthetic audio for testing."""
        np.random.seed(seed)
        sr = settings.SAMPLE_RATE
        duration = settings.AUDIO_DURATION
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)

        # Different harmonic frequencies for each class
        base_freq = 200.0 + class_id * 150.0
        audio = (
            0.5 * np.sin(2 * np.pi * base_freq * t) +
            0.2 * np.sin(2 * np.pi * (base_freq * 1.5) * t) +
            0.05 * np.random.randn(len(t))
        )
        return audio.astype(np.float32), sr

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Retrieve a preprocessed Log-Mel Spectrogram and target label.

        Returns:
            Tuple of (spectrogram, label):
                - spectrogram: torch.FloatTensor of shape (1, n_mels=128, time_frames=173)
                - label: torch.LongTensor scalar with class ID (0-9)
        """
        item = self.samples[idx]

        if item.get("synthetic", False):
            audio, sr = self._get_synthetic_audio(item["seed"], item["class_id"])
        else:
            try:
                audio, sr = load_audio(item["file_path"])
            except Exception as e:
                logger.warning(f"Failed to decode audio file {item['file_path']}: {e}. Using zero-padding.")
                sr = settings.SAMPLE_RATE
                audio = np.zeros(int(sr * settings.AUDIO_DURATION), dtype=np.float32)

        # Preprocessing (peak normalization)
        audio = preprocess_audio(audio, sr)

        # Audio augmentation (training only)
        if self.augmenter is not None:
            audio = self.augmenter(audio, sr)

        # Feature extraction (Log-Mel Spectrogram)
        log_mel = extract_log_mel_spectrogram(audio, sr)

        # Convert to tensor: shape (1, n_mels, time_frames)
        feature_tensor = torch.from_numpy(log_mel).unsqueeze(0).float()
        label_tensor = torch.tensor(item["class_id"], dtype=torch.long)

        return feature_tensor, label_tensor


def create_dataloaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    batch_size: int = 32,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Create PyTorch DataLoaders for train and validation datasets.

    Args:
        train_dataset: Training Dataset.
        val_dataset: Validation Dataset.
        batch_size: Batch size. Defaults to 32.
        num_workers: Number of subprocesses for data loading (0 on Windows recommended).

    Returns:
        Tuple of (train_loader, val_loader).
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader
