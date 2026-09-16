"""
Audio Exploration Script — Phase 1 Milestone

This script demonstrates the complete audio processing pipeline:
1. Load an UrbanSound8K .wav file
2. Display file information
3. Extract all spectral features (spectrogram, mel, log-mel)
4. Print feature shapes and statistics
5. Show a 4-panel visualization
6. Save the visualization to results/

Usage:
    python scripts/explore_audio.py
    python scripts/explore_audio.py path/to/audio.wav
    python scripts/explore_audio.py --find-sample

Expected Output:
    ============================================================
    AUDIO FILE INFORMATION
    ============================================================
    Filename:     <file>.wav
    Duration:     4.00 seconds
    Sample Rate:  22050 Hz
    Num Samples:  88200
    ...
"""

import sys
import logging
from pathlib import Path

# Add project root to Python path so we can import src modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import settings
from src.audio.processor import load_audio, preprocess_audio, get_audio_info
from src.features.extractor import extract_all_features
from src.utils.visualization import plot_all_features

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def find_sample_audio() -> Path | None:
    """Find a sample .wav file.

    Searches:
    1. UrbanSound8K dataset folds (fold1..fold10)
    2. data/sample/
    3. data/raw/

    Returns:
        Path to a .wav file, or None if no audio file is found.
    """
    # 1. Check UrbanSound8K folds
    audio_dir = settings.URBANSOUND8K_AUDIO_DIR
    if audio_dir.exists():
        for fold_num in range(1, 11):
            fold_dir = audio_dir / f"fold{fold_num}"
            if fold_dir.exists():
                wav_files = list(fold_dir.glob("*.wav"))
                if wav_files:
                    return wav_files[0]

    # 2. Check data/sample
    sample_dir = settings.DATA_DIR / "sample"
    if sample_dir.exists():
        sample_wavs = list(sample_dir.glob("*.wav"))
        if sample_wavs:
            return sample_wavs[0]

    # 3. Check data/raw
    raw_dir = settings.RAW_DATA_DIR
    if raw_dir.exists():
        raw_wavs = list(raw_dir.glob("*.wav"))
        if raw_wavs:
            return raw_wavs[0]

    return None


def print_separator(char: str = "=", width: int = 60) -> None:
    """Print a visual separator line."""
    print(char * width)


def explore_audio(file_path: str | Path, show_plot: bool = False) -> None:
    """Run the complete audio exploration pipeline on a single file.

    Args:
        file_path: Path to the audio file to explore.
        show_plot: Whether to display interactive plot window.
    """
    file_path = Path(file_path)

    # ============================================================
    # Step 1: File Information
    # ============================================================
    print()
    print_separator()
    print("AUDIO FILE INFORMATION")
    print_separator()

    info = get_audio_info(file_path)
    print(f"  Filename:       {info['filename']}")
    print(f"  Path:           {info['filepath']}")
    print(f"  Duration:       {info['duration']:.2f} seconds")
    print(f"  Sample Rate:    {info['sample_rate']} Hz")
    print(f"  Num Samples:    {info['num_samples']}")
    print(f"  Num Channels:   {info['num_channels']}")

    # ============================================================
    # Step 2: Load and Preprocess
    # ============================================================
    print_separator("-")
    print("LOADING AND PREPROCESSING")
    print_separator("-")

    audio, sr = load_audio(file_path)
    print(f"  Loaded SR:      {sr} Hz")
    print(f"  Loaded Samples: {len(audio)}")
    print(f"  Loaded Duration:{len(audio) / sr:.2f} seconds")

    audio = preprocess_audio(audio, sr)
    print("  After Preprocess:")
    print(f"    Min:          {audio.min():.4f}")
    print(f"    Max:          {audio.max():.4f}")
    print(f"    Mean:         {audio.mean():.4f}")
    print(f"    Std:          {audio.std():.4f}")

    # ============================================================
    # Step 3: Feature Extraction
    # ============================================================
    print_separator("-")
    print("EXTRACTED FEATURES")
    print_separator("-")

    features = extract_all_features(audio, sr)

    print(f"  Spectrogram shape:      {features['spectrogram'].shape}")
    print(f"    -> Frequency bins:    {features['spectrogram'].shape[0]}")
    print(f"    -> Time frames:       {features['spectrogram'].shape[1]}")
    print()
    print(f"  Mel Spectrogram shape:  {features['mel_spectrogram'].shape}")
    print(f"    -> Mel bands:         {features['mel_spectrogram'].shape[0]}")
    print(f"    -> Time frames:       {features['mel_spectrogram'].shape[1]}")
    print()
    print(f"  Log-Mel Spectrogram:    {features['log_mel_spectrogram'].shape}")
    print(f"    -> Mel bands:         {features['log_mel_spectrogram'].shape[0]}")
    print(f"    -> Time frames:       {features['log_mel_spectrogram'].shape[1]}")
    print(f"    -> Min value:         {features['log_mel_spectrogram'].min():.1f} dB")
    print(f"    -> Max value:         {features['log_mel_spectrogram'].max():.1f} dB")

    # ============================================================
    # Step 4: Configuration Summary
    # ============================================================
    print_separator("-")
    print("CONFIGURATION")
    print_separator("-")
    print(f"  Sample Rate:    {settings.SAMPLE_RATE} Hz")
    print(f"  Audio Duration: {settings.AUDIO_DURATION} s")
    print(f"  N_FFT:          {settings.N_FFT}")
    print(f"  Hop Length:     {settings.HOP_LENGTH}")
    print(f"  N_MELS:         {settings.N_MELS}")

    print_separator()
    print()

    # ============================================================
    # Step 5: Visualization
    # ============================================================
    title = f"Audio Features - {info['filename']}"
    save_path = settings.RESULTS_DIR / f"explore_{file_path.stem}.png"

    logger.info("Generating 4-panel feature visualization...")
    plot_all_features(
        audio, sr, features,
        title=title,
        save_path=save_path,
        show=show_plot,
    )

    logger.info(f"Visualization saved to: {save_path}")
    print()
    print(f"[SUCCESS] Exploration complete! Plot saved to: {save_path}")
    print()


def main():
    """Entry point for the exploration script."""
    show_plot = "--show" in sys.argv
    args = [arg for arg in sys.argv[1:] if arg != "--show"]

    # Determine which audio file to use
    if len(args) > 0:
        if args[0] == "--find-sample":
            file_path = find_sample_audio()
            if file_path is None:
                print("[ERROR] No audio file found!")
                print(f"   Expected location: {settings.URBANSOUND8K_AUDIO_DIR} or {settings.DATA_DIR / 'sample'}")
                print("   Please run: python scripts/generate_sample_audio.py")
                print("   Or download UrbanSound8K (see: data/raw/README.md)")
                sys.exit(1)
            print(f"[INFO] Auto-selected sample: {file_path}")
        else:
            file_path = Path(args[0])
    else:
        # Default: try to find a sample in the dataset or sample dir
        file_path = find_sample_audio()
        if file_path is None:
            print("[ERROR] No audio file specified and no sample audio found!")
            print()
            print("Usage:")
            print("  python scripts/explore_audio.py path/to/audio.wav")
            print("  python scripts/explore_audio.py --find-sample")
            print()
            print("To generate a sample urban audio clip for testing, run:")
            print("  python scripts/generate_sample_audio.py")
            print()
            print("Or place the UrbanSound8K dataset at:")
            print(f"  {settings.URBANSOUND8K_AUDIO_DIR}")
            print()
            print("See data/raw/README.md for download instructions.")
            sys.exit(1)
        print(f"[INFO] Auto-selected sample: {file_path}")

    if not file_path.exists():
        print(f"[ERROR] File not found: {file_path}")
        sys.exit(1)

    explore_audio(file_path, show_plot=show_plot)


if __name__ == "__main__":
    main()
