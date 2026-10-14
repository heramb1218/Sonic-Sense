"""
Audio Augmentation — Configurable data augmentation techniques for audio.

Techniques implemented:
- Time shifting (rolling the audio along time axis)
- Noise injection (adding low-level Gaussian white noise)
- Pitch shifting (shifting pitch by semitones)
- Time stretching (speeding up or slowing down without changing pitch)

Usage:
    from src.audio.augmentation import AudioAugmenter

    augmenter = AudioAugmenter(use_shift=True, use_noise=True)
    aug_audio = augmenter(audio, sr)
"""

import logging
from typing import Optional
import numpy as np
import librosa

logger = logging.getLogger(__name__)


def time_shift(audio: np.ndarray, shift_limit: float = 0.2) -> np.ndarray:
    """Shift audio horizontally along time axis (circular roll).

    Args:
        audio: 1D numpy array of audio samples.
        shift_limit: Maximum fraction of total length to shift (e.g. 0.2 = ±20%).

    Returns:
        Shifted audio array.
    """
    shift_max = int(len(audio) * shift_limit)
    if shift_max == 0:
        return audio
    shift = np.random.randint(-shift_max, shift_max)
    return np.roll(audio, shift)


def add_noise(audio: np.ndarray, noise_factor: float = 0.01) -> np.ndarray:
    """Add Gaussian white noise to the audio.

    Args:
        audio: 1D numpy array of audio samples.
        noise_factor: Relative noise amplitude.

    Returns:
        Audio with injected noise.
    """
    noise = np.random.randn(len(audio))
    return audio + noise_factor * noise.astype(audio.dtype)


def pitch_shift(audio: np.ndarray, sr: int, n_steps: Optional[float] = None) -> np.ndarray:
    """Shift pitch of the audio by semitones.

    Args:
        audio: 1D numpy array of audio samples.
        sr: Sample rate in Hz.
        n_steps: Number of semitones to shift. If None, chosen uniformly from [-2.0, 2.0].

    Returns:
        Pitch-shifted audio array.
    """
    if n_steps is None:
        n_steps = float(np.random.uniform(-2.0, 2.0))
    if abs(n_steps) < 0.1:
        return audio
    try:
        return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
    except Exception as e:
        logger.warning(f"Pitch shift failed, returning original audio: {e}")
        return audio


def time_stretch(audio: np.ndarray, rate: Optional[float] = None) -> np.ndarray:
    """Stretch time of audio without changing pitch, preserving length by padding/trimming.

    Args:
        audio: 1D numpy array of audio samples.
        rate: Stretch factor (>1.0 speeds up, <1.0 slows down).
            If None, chosen uniformly from [0.9, 1.1].

    Returns:
        Length-preserved time-stretched audio array.
    """
    if rate is None:
        rate = float(np.random.uniform(0.9, 1.1))
    if abs(rate - 1.0) < 0.02:
        return audio
    try:
        stretched = librosa.effects.time_stretch(audio, rate=rate)
        # Preserve original length
        target_len = len(audio)
        if len(stretched) < target_len:
            return np.pad(stretched, (0, target_len - len(stretched)), mode="constant")
        else:
            return stretched[:target_len]
    except Exception as e:
        logger.warning(f"Time stretch failed, returning original audio: {e}")
        return audio


class AudioAugmenter:
    """Configurable audio augmentation pipeline.

    Allows toggling each augmentation on/off and setting probability.
    """

    def __init__(
        self,
        use_shift: bool = True,
        use_noise: bool = True,
        use_pitch: bool = False,
        use_stretch: bool = False,
        prob: float = 0.5,
    ):
        self.use_shift = use_shift
        self.use_noise = use_noise
        self.use_pitch = use_pitch
        self.use_stretch = use_stretch
        self.prob = prob

    def __call__(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Apply active augmentations with probability self.prob."""
        aug_audio = audio.copy()

        if self.use_shift and np.random.rand() < self.prob:
            aug_audio = time_shift(aug_audio)

        if self.use_noise and np.random.rand() < self.prob:
            aug_audio = add_noise(aug_audio)

        if self.use_pitch and np.random.rand() < self.prob:
            aug_audio = pitch_shift(aug_audio, sr=sr)

        if self.use_stretch and np.random.rand() < self.prob:
            aug_audio = time_stretch(aug_audio)

        return aug_audio
