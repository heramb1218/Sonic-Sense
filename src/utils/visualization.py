"""
Visualization Utilities — Audio and Feature Plotting

Provides functions to visualize waveforms, spectrograms, and Mel
spectrograms. Used in exploration scripts and notebooks.

Usage:
    from src.utils.visualization import plot_waveform, plot_all_features

    plot_waveform(audio, sr, title="My Audio")
    plot_all_features(audio, sr, features_dict, title="All Features")
"""

import logging
from pathlib import Path
from typing import Optional

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

from src.config.settings import settings

logger = logging.getLogger(__name__)


def plot_waveform(
    audio: np.ndarray,
    sr: int,
    title: str = "Waveform",
    figsize: tuple[int, int] = (12, 3),
    save_path: Optional[str | Path] = None,
) -> None:
    """Plot the time-domain waveform of an audio signal.

    Args:
        audio: 1D numpy array of audio samples.
        sr: Sample rate in Hz.
        title: Plot title.
        figsize: Figure size (width, height) in inches.
        save_path: If provided, save the figure to this path.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    librosa.display.waveshow(audio, sr=sr, ax=ax, color="#4A90D9")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        _save_figure(fig, save_path)
    plt.show()


def plot_spectrogram(
    spectrogram: np.ndarray,
    sr: int,
    hop_length: Optional[int] = None,
    title: str = "Spectrogram",
    figsize: tuple[int, int] = (12, 4),
    save_path: Optional[str | Path] = None,
) -> None:
    """Plot a magnitude spectrogram.

    Args:
        spectrogram: 2D numpy array (frequency_bins, time_frames).
        sr: Sample rate in Hz.
        hop_length: Hop length used in STFT. Defaults to settings.HOP_LENGTH.
        title: Plot title.
        figsize: Figure size.
        save_path: If provided, save the figure.
    """
    if hop_length is None:
        hop_length = settings.HOP_LENGTH

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Convert to dB for better visualization
    spec_db = librosa.amplitude_to_db(spectrogram, ref=np.max)
    img = librosa.display.specshow(
        spec_db, sr=sr, hop_length=hop_length,
        x_axis="time", y_axis="hz", ax=ax, cmap="magma"
    )
    ax.set_title(title, fontsize=14, fontweight="bold")
    fig.colorbar(img, ax=ax, format="%+2.0f dB", label="Amplitude (dB)")
    plt.tight_layout()

    if save_path:
        _save_figure(fig, save_path)
    plt.show()


def plot_mel_spectrogram(
    mel_spectrogram: np.ndarray,
    sr: int,
    hop_length: Optional[int] = None,
    title: str = "Mel Spectrogram",
    figsize: tuple[int, int] = (12, 4),
    is_db: bool = False,
    save_path: Optional[str | Path] = None,
) -> None:
    """Plot a Mel spectrogram.

    Args:
        mel_spectrogram: 2D numpy array (n_mels, time_frames).
        sr: Sample rate in Hz.
        hop_length: Hop length. Defaults to settings.HOP_LENGTH.
        title: Plot title.
        figsize: Figure size.
        is_db: If True, data is already in dB scale (skip conversion).
        save_path: If provided, save the figure.
    """
    if hop_length is None:
        hop_length = settings.HOP_LENGTH

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    if not is_db:
        display_data = librosa.power_to_db(mel_spectrogram, ref=np.max)
    else:
        display_data = mel_spectrogram

    img = librosa.display.specshow(
        display_data, sr=sr, hop_length=hop_length,
        x_axis="time", y_axis="mel", ax=ax, cmap="magma"
    )
    ax.set_title(title, fontsize=14, fontweight="bold")
    fig.colorbar(img, ax=ax, format="%+2.0f dB", label="Power (dB)")
    plt.tight_layout()

    if save_path:
        _save_figure(fig, save_path)
    plt.show()


def plot_log_mel_spectrogram(
    log_mel: np.ndarray,
    sr: int,
    hop_length: Optional[int] = None,
    title: str = "Log-Mel Spectrogram",
    figsize: tuple[int, int] = (12, 4),
    save_path: Optional[str | Path] = None,
) -> None:
    """Plot a Log-Mel spectrogram (already in dB scale).

    Args:
        log_mel: 2D numpy array (n_mels, time_frames) in dB.
        sr: Sample rate in Hz.
        hop_length: Hop length. Defaults to settings.HOP_LENGTH.
        title: Plot title.
        figsize: Figure size.
        save_path: If provided, save the figure.
    """
    # Log-Mel is already in dB, so use is_db=True
    plot_mel_spectrogram(
        log_mel, sr, hop_length=hop_length,
        title=title, figsize=figsize, is_db=True,
        save_path=save_path,
    )


def plot_all_features(
    audio: np.ndarray,
    sr: int,
    features: dict[str, np.ndarray],
    title: str = "Audio Features",
    hop_length: Optional[int] = None,
    figsize: tuple[int, int] = (14, 12),
    save_path: Optional[str | Path] = None,
    show: bool = True,
) -> None:
    """Plot a combined 4-panel figure: waveform + spectrogram + mel + log-mel.

    This is the main visualization function for audio exploration.

    Args:
        audio: 1D numpy array of audio samples.
        sr: Sample rate in Hz.
        features: Dictionary with keys "spectrogram", "mel_spectrogram",
            and "log_mel_spectrogram".
        title: Overall figure title.
        hop_length: Hop length. Defaults to settings.HOP_LENGTH.
        figsize: Figure size.
        save_path: If provided, save the figure.
        show: If True, call plt.show(). Default True.
    """
    if hop_length is None:
        hop_length = settings.HOP_LENGTH

    fig, axes = plt.subplots(4, 1, figsize=figsize)
    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.01)

    # Panel 1: Waveform
    librosa.display.waveshow(audio, sr=sr, ax=axes[0], color="#4A90D9")
    axes[0].set_title("Waveform", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("")  # Remove x-label to reduce clutter
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True, alpha=0.3)

    # Panel 2: Spectrogram
    spec_db = librosa.amplitude_to_db(features["spectrogram"], ref=np.max)
    img1 = librosa.display.specshow(
        spec_db, sr=sr, hop_length=hop_length,
        x_axis="time", y_axis="hz", ax=axes[1], cmap="magma"
    )
    axes[1].set_title("Spectrogram (STFT)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("")
    fig.colorbar(img1, ax=axes[1], format="%+2.0f dB")

    # Panel 3: Mel Spectrogram
    mel_db = librosa.power_to_db(features["mel_spectrogram"], ref=np.max)
    img2 = librosa.display.specshow(
        mel_db, sr=sr, hop_length=hop_length,
        x_axis="time", y_axis="mel", ax=axes[2], cmap="magma"
    )
    axes[2].set_title("Mel Spectrogram", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("")
    fig.colorbar(img2, ax=axes[2], format="%+2.0f dB")

    # Panel 4: Log-Mel Spectrogram
    img3 = librosa.display.specshow(
        features["log_mel_spectrogram"], sr=sr, hop_length=hop_length,
        x_axis="time", y_axis="mel", ax=axes[3], cmap="magma"
    )
    axes[3].set_title("Log-Mel Spectrogram (Primary ML Feature)", fontsize=12, fontweight="bold")
    fig.colorbar(img3, ax=axes[3], format="%+2.0f dB")

    plt.tight_layout()

    if save_path:
        _save_figure(fig, save_path)
    if show:
        plt.show()
    plt.close(fig)


def _save_figure(fig: plt.Figure, save_path: str | Path) -> None:
    """Save a matplotlib figure to disk.

    Creates parent directories if they don't exist.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    logger.info(f"Figure saved: {save_path}")
