"""
Feature Extractor — Spectrogram and Mel Feature Extraction

Converts raw audio signals into spectral representations suitable
for machine learning models. The primary feature used throughout
this project is the Log-Mel Spectrogram.

Usage:
    from src.features.extractor import (
        extract_spectrogram,
        extract_mel_spectrogram,
        extract_log_mel_spectrogram,
        save_features,
        load_features,
    )

    spec = extract_spectrogram(audio, sr)
    mel = extract_mel_spectrogram(audio, sr)
    log_mel = extract_log_mel_spectrogram(audio, sr)
    save_features(log_mel, "output.npy")
"""

import logging
from pathlib import Path
from typing import Optional, Callable

import librosa
import numpy as np

from src.config.settings import settings

logger = logging.getLogger(__name__)


def extract_spectrogram(
    audio: np.ndarray,
    sr: int,
    n_fft: Optional[int] = None,
    hop_length: Optional[int] = None,
) -> np.ndarray:
    """Compute the magnitude spectrogram using Short-Time Fourier Transform (STFT).

    The STFT decomposes the audio signal into overlapping windowed segments,
    applies FFT to each, and returns the magnitude of the frequency components.

    Args:
        audio: 1D numpy array of audio samples.
        sr: Sample rate in Hz (used for documentation; not directly in computation).
        n_fft: FFT window size. Defaults to settings.N_FFT (2048).
            Determines frequency resolution: freq_bins = n_fft/2 + 1.
        hop_length: Hop between successive windows. Defaults to settings.HOP_LENGTH (512).
            Determines time resolution: time_frames = 1 + num_samples / hop_length.

    Returns:
        2D numpy array of shape (1 + n_fft/2, time_frames) containing
        magnitude values.
    """
    if n_fft is None:
        n_fft = settings.N_FFT
    if hop_length is None:
        hop_length = settings.HOP_LENGTH

    # Compute STFT and take magnitude
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    spectrogram = np.abs(stft)

    logger.debug(
        f"Spectrogram: n_fft={n_fft}, hop_length={hop_length}, "
        f"shape={spectrogram.shape}"
    )

    return spectrogram


def extract_mel_spectrogram(
    audio: np.ndarray,
    sr: int,
    n_fft: Optional[int] = None,
    hop_length: Optional[int] = None,
    n_mels: Optional[int] = None,
) -> np.ndarray:
    """Compute the Mel spectrogram.

    The Mel spectrogram applies a bank of triangular filters spaced according
    to the Mel scale (which approximates human pitch perception) to the
    power spectrogram. This reduces the number of frequency bins and
    emphasizes perceptually relevant frequencies.

    Args:
        audio: 1D numpy array of audio samples.
        sr: Sample rate in Hz.
        n_fft: FFT window size. Defaults to settings.N_FFT (2048).
        hop_length: Hop length. Defaults to settings.HOP_LENGTH (512).
        n_mels: Number of Mel filter banks. Defaults to settings.N_MELS (128).

    Returns:
        2D numpy array of shape (n_mels, time_frames) containing
        Mel-scaled power values.
    """
    if n_fft is None:
        n_fft = settings.N_FFT
    if hop_length is None:
        hop_length = settings.HOP_LENGTH
    if n_mels is None:
        n_mels = settings.N_MELS

    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    )

    logger.debug(
        f"Mel Spectrogram: n_mels={n_mels}, n_fft={n_fft}, "
        f"hop_length={hop_length}, shape={mel_spec.shape}"
    )

    return mel_spec


def extract_log_mel_spectrogram(
    audio: np.ndarray,
    sr: int,
    n_fft: Optional[int] = None,
    hop_length: Optional[int] = None,
    n_mels: Optional[int] = None,
    ref: float | Callable = np.max,
    top_db: Optional[float] = 80.0,
) -> np.ndarray:
    """Compute the Log-Mel spectrogram (primary feature for ML models).

    This is the standard audio representation used for CNN-based sound
    classification. It applies a logarithmic (dB) scaling to the Mel
    spectrogram, which:
    - Compresses the dynamic range (similar to human loudness perception)
    - Makes features more suitable for neural network training
    - Produces a 2D "image" that CNNs can process effectively

    Args:
        audio: 1D numpy array of audio samples.
        sr: Sample rate in Hz.
        n_fft: FFT window size. Defaults to settings.N_FFT (2048).
        hop_length: Hop length. Defaults to settings.HOP_LENGTH (512).
        n_mels: Number of Mel filter banks. Defaults to settings.N_MELS (128).
        ref: Reference power for dB conversion. Defaults to np.max (peak-normalized to 0 dB).
        top_db: Threshold below the peak to clip. Default 80.0 dB.

    Returns:
        2D numpy array of shape (n_mels, time_frames) containing
        log-scaled (dB) Mel spectrogram values.
    """
    # First compute the Mel spectrogram
    mel_spec = extract_mel_spectrogram(
        audio, sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )

    # Convert power to decibels (log scale)
    log_mel = librosa.power_to_db(mel_spec, ref=ref, top_db=top_db)

    logger.debug(
        f"Log-Mel Spectrogram: shape={log_mel.shape}, "
        f"min={log_mel.min():.1f} dB, max={log_mel.max():.1f} dB"
    )

    return log_mel


def extract_all_features(
    audio: np.ndarray,
    sr: int,
    n_fft: Optional[int] = None,
    hop_length: Optional[int] = None,
    n_mels: Optional[int] = None,
) -> dict[str, np.ndarray]:
    """Extract all spectral features from an audio signal.

    Convenience function that computes all three spectral representations
    in one call.

    Args:
        audio: 1D numpy array of audio samples.
        sr: Sample rate in Hz.
        n_fft: FFT window size. Defaults to settings.N_FFT.
        hop_length: Hop length. Defaults to settings.HOP_LENGTH.
        n_mels: Number of Mel bands. Defaults to settings.N_MELS.

    Returns:
        Dictionary with keys:
            - "spectrogram": Magnitude spectrogram
            - "mel_spectrogram": Mel spectrogram
            - "log_mel_spectrogram": Log-Mel spectrogram
    """
    spectrogram = extract_spectrogram(audio, sr, n_fft=n_fft, hop_length=hop_length)
    mel_spec = extract_mel_spectrogram(
        audio, sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    log_mel = extract_log_mel_spectrogram(
        audio, sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )

    return {
        "spectrogram": spectrogram,
        "mel_spectrogram": mel_spec,
        "log_mel_spectrogram": log_mel,
    }


def save_features(features: np.ndarray, output_path: str | Path) -> Path:
    """Save extracted features to a .npy file.

    Args:
        features: Numpy array of features to save.
        output_path: Destination file path (should end with .npy).

    Returns:
        Path to the saved file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(output_path), features)
    logger.info(f"Features saved: {output_path} (shape={features.shape})")
    return output_path


def load_features(input_path: str | Path) -> np.ndarray:
    """Load features from a .npy file.

    Args:
        input_path: Path to the .npy file.

    Returns:
        Numpy array of features.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Feature file not found: {input_path}")
    features = np.load(str(input_path))
    logger.debug(f"Features loaded: {input_path} (shape={features.shape})")
    return features
