"""
Sample Audio Generator — Generates synthetic urban sound samples for local testing.

Creates realistic synthetic audio files (e.g., urban siren, traffic rumble)
so that the audio processing, feature extraction, and ML pipeline can be tested
immediately even before the full 6GB UrbanSound8K dataset is downloaded.

Usage:
    python scripts/generate_sample_audio.py
"""

import sys
from pathlib import Path
import numpy as np
import soundfile as sf

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import settings


def generate_synthetic_siren(
    duration: float = 4.0,
    sr: int = 22050,
    base_freq: float = 750.0,
    mod_depth: float = 250.0,
    mod_rate: float = 0.5,
) -> np.ndarray:
    """Generate a synthetic emergency siren sound with harmonics and background noise.

    Args:
        duration: Duration in seconds.
        sr: Sample rate in Hz.
        base_freq: Center frequency of the siren (Hz).
        mod_depth: Modulation depth (Hz).
        mod_rate: Frequency of modulation (Hz).

    Returns:
        1D numpy array of audio samples.
    """
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    # Frequency modulation: center frequency sweeps up and down
    freq_inst = base_freq + mod_depth * np.sin(2 * np.pi * mod_rate * t)
    phase = 2 * np.pi * np.cumsum(freq_inst) / sr

    # Fundamental + harmonics
    siren = (
        0.5 * np.sin(phase) +
        0.25 * np.sin(2 * phase) +
        0.12 * np.sin(3 * phase)
    )

    # Add low-level ambient street noise
    noise = np.random.normal(0, 0.03, size=len(t))

    signal = siren + noise
    # Normalize to [-0.95, 0.95]
    signal = 0.95 * signal / np.max(np.abs(signal))
    return signal.astype(np.float32)


def main():
    """Generate sample audio files and save to data/sample/."""
    sample_dir = settings.DATA_DIR / "sample"
    sample_dir.mkdir(parents=True, exist_ok=True)

    siren_path = sample_dir / "sample_siren_urban.wav"
    audio = generate_synthetic_siren(
        duration=settings.AUDIO_DURATION,
        sr=settings.SAMPLE_RATE,
    )

    sf.write(str(siren_path), audio, settings.SAMPLE_RATE)
    print(f"[SUCCESS] Generated sample audio: {siren_path}")
    print(f"   Duration: {len(audio)/settings.SAMPLE_RATE:.2f}s | Sample Rate: {settings.SAMPLE_RATE} Hz | Samples: {len(audio)}")


if __name__ == "__main__":
    main()
