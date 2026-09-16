"""
Tests for Audio Processing and Feature Extraction — Phase 1

Tests verify:
- Audio loading (correct length, sample rate)
- Preprocessing (normalization)
- Feature extraction (correct shapes)
- Audio info extraction

These tests use a synthetically generated audio signal so they
do NOT require the UrbanSound8K dataset to be present.

Usage:
    pytest tests/test_audio.py -v
"""

import sys
from pathlib import Path
import tempfile

import numpy as np
import pytest
import soundfile as sf

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import settings
from src.audio.processor import load_audio, preprocess_audio, get_audio_info
from src.features.extractor import (
    extract_spectrogram,
    extract_mel_spectrogram,
    extract_log_mel_spectrogram,
    extract_all_features,
    save_features,
    load_features,
)


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def sample_wav(tmp_path):
    """Create a temporary WAV file with a synthetic sine wave.

    Generates a 2-second, 22050 Hz mono sine wave at 440 Hz (A4 note).
    This allows testing without needing the actual dataset.
    """
    sr = 22050
    duration = 2.0  # seconds (shorter than 4s to test padding)
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # 440 Hz sine wave with some harmonics for a richer spectrum
    audio = (
        0.5 * np.sin(2 * np.pi * 440 * t) +
        0.3 * np.sin(2 * np.pi * 880 * t) +
        0.1 * np.sin(2 * np.pi * 1320 * t)
    )
    audio = audio.astype(np.float32)

    wav_path = tmp_path / "test_sine.wav"
    sf.write(str(wav_path), audio, sr)
    return wav_path


@pytest.fixture
def sample_audio():
    """Create a synthetic audio numpy array (4 seconds, 22050 Hz)."""
    sr = settings.SAMPLE_RATE
    duration = settings.AUDIO_DURATION
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    return audio, sr


# ============================================================
# Audio Loading Tests
# ============================================================

class TestLoadAudio:
    """Tests for the load_audio function."""

    def test_load_audio_returns_correct_length(self, sample_wav):
        """Audio should be padded/truncated to exactly AUDIO_DURATION * SAMPLE_RATE samples."""
        audio, sr = load_audio(sample_wav)
        expected_length = int(settings.SAMPLE_RATE * settings.AUDIO_DURATION)
        assert len(audio) == expected_length, (
            f"Expected {expected_length} samples, got {len(audio)}"
        )

    def test_load_audio_returns_correct_sample_rate(self, sample_wav):
        """Sample rate should match the configured SAMPLE_RATE."""
        audio, sr = load_audio(sample_wav)
        assert sr == settings.SAMPLE_RATE

    def test_load_audio_returns_1d_array(self, sample_wav):
        """Mono audio should be a 1D numpy array."""
        audio, sr = load_audio(sample_wav)
        assert audio.ndim == 1

    def test_load_audio_pads_short_clips(self, sample_wav):
        """A 2-second clip should be zero-padded to 4 seconds."""
        audio, sr = load_audio(sample_wav)
        expected_length = int(settings.SAMPLE_RATE * settings.AUDIO_DURATION)
        assert len(audio) == expected_length
        # The tail should be zeros (padding)
        tail = audio[-int(sr * 1.5):]  # Last 1.5 seconds should be mostly zeros
        assert np.allclose(tail, 0, atol=1e-6), "Padding should be zeros"

    def test_load_audio_file_not_found(self):
        """Should raise FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError):
            load_audio("nonexistent_file.wav")


# ============================================================
# Preprocessing Tests
# ============================================================

class TestPreprocessAudio:
    """Tests for the preprocess_audio function."""

    def test_normalization_scales_to_unit_range(self):
        """After normalization, the peak should be 1.0."""
        audio = np.array([0.0, 0.5, -0.3, 0.2], dtype=np.float32)
        processed = preprocess_audio(audio, sr=22050, normalize=True)
        assert np.isclose(np.max(np.abs(processed)), 1.0), (
            f"Peak should be 1.0 after normalization, got {np.max(np.abs(processed))}"
        )

    def test_normalization_preserves_silent_audio(self):
        """Silent audio (all zeros) should remain all zeros."""
        audio = np.zeros(1000, dtype=np.float32)
        processed = preprocess_audio(audio, sr=22050, normalize=True)
        assert np.allclose(processed, 0)

    def test_no_normalization(self):
        """With normalize=False, audio should be unchanged."""
        audio = np.array([0.0, 0.5, -0.3], dtype=np.float32)
        processed = preprocess_audio(audio, sr=22050, normalize=False)
        assert np.array_equal(audio, processed)


# ============================================================
# Feature Extraction Tests
# ============================================================

class TestFeatureExtraction:
    """Tests for spectral feature extraction functions."""

    def test_spectrogram_shape(self, sample_audio):
        """Spectrogram shape should be (1 + n_fft/2, time_frames)."""
        audio, sr = sample_audio
        spec = extract_spectrogram(audio, sr)
        expected_freq_bins = 1 + settings.N_FFT // 2
        assert spec.shape[0] == expected_freq_bins, (
            f"Expected {expected_freq_bins} frequency bins, got {spec.shape[0]}"
        )
        # Time frames should be positive
        assert spec.shape[1] > 0

    def test_mel_spectrogram_shape(self, sample_audio):
        """Mel spectrogram shape should be (n_mels, time_frames)."""
        audio, sr = sample_audio
        mel = extract_mel_spectrogram(audio, sr)
        assert mel.shape[0] == settings.N_MELS, (
            f"Expected {settings.N_MELS} mel bands, got {mel.shape[0]}"
        )
        assert mel.shape[1] > 0

    def test_log_mel_spectrogram_shape(self, sample_audio):
        """Log-Mel spectrogram shape should be (n_mels, time_frames)."""
        audio, sr = sample_audio
        log_mel = extract_log_mel_spectrogram(audio, sr)
        assert log_mel.shape[0] == settings.N_MELS
        assert log_mel.shape[1] > 0

    def test_log_mel_values_in_db(self, sample_audio):
        """Log-Mel values should be in dB (negative or zero, clipped at -80)."""
        audio, sr = sample_audio
        log_mel = extract_log_mel_spectrogram(audio, sr)
        assert log_mel.max() <= 0.1, "Max dB should be near 0"
        assert log_mel.min() >= -80.1, "Min dB should be >= -80"

    def test_extract_all_features_keys(self, sample_audio):
        """extract_all_features should return a dict with 3 expected keys."""
        audio, sr = sample_audio
        features = extract_all_features(audio, sr)
        expected_keys = {"spectrogram", "mel_spectrogram", "log_mel_spectrogram"}
        assert set(features.keys()) == expected_keys

    def test_mel_and_log_mel_same_shape(self, sample_audio):
        """Mel and Log-Mel spectrograms should have the same shape."""
        audio, sr = sample_audio
        mel = extract_mel_spectrogram(audio, sr)
        log_mel = extract_log_mel_spectrogram(audio, sr)
        assert mel.shape == log_mel.shape


# ============================================================
# Feature Save/Load Tests
# ============================================================

class TestFeatureSaveLoad:
    """Tests for saving and loading feature arrays."""

    def test_save_and_load_roundtrip(self, sample_audio, tmp_path):
        """Saved features should be loadable with identical values."""
        audio, sr = sample_audio
        log_mel = extract_log_mel_spectrogram(audio, sr)

        output_path = tmp_path / "test_features.npy"
        save_features(log_mel, output_path)

        loaded = load_features(output_path)
        assert np.array_equal(log_mel, loaded)

    def test_load_nonexistent_file(self):
        """Should raise FileNotFoundError for missing feature files."""
        with pytest.raises(FileNotFoundError):
            load_features("nonexistent.npy")


# ============================================================
# Audio Info Tests
# ============================================================

class TestAudioInfo:
    """Tests for get_audio_info function."""

    def test_audio_info_keys(self, sample_wav):
        """get_audio_info should return expected metadata keys."""
        info = get_audio_info(sample_wav)
        expected_keys = {"filename", "filepath", "duration", "sample_rate",
                         "num_samples", "num_channels"}
        assert set(info.keys()) == expected_keys

    def test_audio_info_values(self, sample_wav):
        """Metadata values should be reasonable for a 2-second sine wave."""
        info = get_audio_info(sample_wav)
        assert info["filename"] == "test_sine.wav"
        assert abs(info["duration"] - 2.0) < 0.1  # ~2 seconds
        assert info["sample_rate"] == 22050
        assert info["num_channels"] == 1
        assert info["num_samples"] > 0

    def test_audio_info_file_not_found(self):
        """Should raise FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError):
            get_audio_info("nonexistent.wav")
