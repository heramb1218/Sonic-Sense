"""
Audio Processor — Loading and Preprocessing

Handles loading audio files, resampling, normalization, and duration
standardization. This module is the entry point for all audio data
in the pipeline.

Usage:
    from src.audio.processor import load_audio, preprocess_audio, get_audio_info

    audio, sr = load_audio("path/to/file.wav")
    audio = preprocess_audio(audio, sr)
    info = get_audio_info("path/to/file.wav")
"""

import logging
from pathlib import Path
from typing import Optional

import librosa
import numpy as np

from src.config.settings import settings

logger = logging.getLogger(__name__)


def load_audio(
    file_path: str | Path,
    sr: Optional[int] = None,
    duration: Optional[float] = None,
    mono: bool = True,
) -> tuple[np.ndarray, int]:
    """Load an audio file and resample to a consistent sample rate.

    Audio clips shorter than `duration` are zero-padded on the right.
    Clips longer than `duration` are truncated.

    Args:
        file_path: Path to the audio file (.wav, .mp3, etc.).
        sr: Target sample rate in Hz. Defaults to settings.SAMPLE_RATE (22050).
        duration: Target duration in seconds. Defaults to settings.AUDIO_DURATION (4.0).
            Pass None to load the full clip without padding/truncation.
        mono: If True, convert to mono. Default True.

    Returns:
        Tuple of (audio_signal, sample_rate).
        audio_signal is a 1D numpy array of shape (num_samples,).

    Raises:
        FileNotFoundError: If the audio file does not exist.
        RuntimeError: If the audio file cannot be loaded.
    """
    file_path = Path(file_path)
    if sr is None:
        sr = settings.SAMPLE_RATE
    if duration is None:
        duration = settings.AUDIO_DURATION

    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    try:
        # librosa.load automatically resamples to `sr`
        audio, loaded_sr = librosa.load(
            str(file_path),
            sr=sr,
            mono=mono,
            duration=duration,
        )
        logger.debug(
            f"Loaded: {file_path.name} | "
            f"SR={loaded_sr} Hz | "
            f"Samples={len(audio)} | "
            f"Duration={len(audio)/loaded_sr:.2f}s"
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load audio file '{file_path}': {e}") from e

    # Ensure consistent length by padding/truncating
    if duration is not None:
        target_length = int(sr * duration)
        audio = _ensure_length(audio, target_length)

    return audio, sr


def preprocess_audio(
    audio: np.ndarray,
    sr: int,
    normalize: bool = True,
) -> np.ndarray:
    """Preprocess an audio signal.

    Steps:
        1. Peak normalization (scale to [-1, 1])

    Args:
        audio: 1D numpy array of audio samples.
        sr: Sample rate (used for potential future processing steps).
        normalize: Whether to apply peak normalization. Default True.

    Returns:
        Preprocessed audio signal as a 1D numpy array.
    """
    if normalize:
        audio = _normalize(audio)

    return audio


def get_audio_info(file_path: str | Path) -> dict:
    """Get metadata about an audio file without loading the full waveform.

    Args:
        file_path: Path to the audio file.

    Returns:
        Dictionary with keys:
            - filename (str): Name of the file
            - filepath (str): Full path to the file
            - duration (float): Duration in seconds
            - sample_rate (int): Native sample rate
            - num_samples (int): Total number of samples
            - num_channels (int): Number of audio channels
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    try:
        import soundfile as sf

        info = sf.info(str(file_path))
        return {
            "filename": file_path.name,
            "filepath": str(file_path),
            "duration": info.duration,
            "sample_rate": info.samplerate,
            "num_samples": info.frames,
            "num_channels": info.channels,
        }
    except Exception as e:
        # Fallback: load with librosa to get info (slower but more compatible)
        logger.warning(f"soundfile.info failed, falling back to librosa: {e}")
        audio, sr = librosa.load(str(file_path), sr=None, mono=False)
        if audio.ndim == 1:
            num_channels = 1
            num_samples = len(audio)
        else:
            num_channels = audio.shape[0]
            num_samples = audio.shape[1]

        return {
            "filename": file_path.name,
            "filepath": str(file_path),
            "duration": num_samples / sr,
            "sample_rate": sr,
            "num_samples": num_samples,
            "num_channels": num_channels,
        }


def _normalize(audio: np.ndarray) -> np.ndarray:
    """Peak-normalize audio to the range [-1, 1].

    If the signal is silent (all zeros), it is returned unchanged.
    """
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak
    return audio


def _ensure_length(audio: np.ndarray, target_length: int) -> np.ndarray:
    """Pad or truncate audio to exactly target_length samples.

    Shorter signals are zero-padded on the right.
    Longer signals are truncated from the right.
    """
    current_length = len(audio)
    if current_length < target_length:
        # Zero-pad on the right
        padding = target_length - current_length
        audio = np.pad(audio, (0, padding), mode="constant", constant_values=0)
        logger.debug(f"Padded audio: {current_length} → {target_length} samples")
    elif current_length > target_length:
        # Truncate from the right
        audio = audio[:target_length]
        logger.debug(f"Truncated audio: {current_length} → {target_length} samples")
    return audio
