"""
Audio Classification Inference Script — Phase 2 & Phase 2.1 Provenance Safety

Classifies an audio file using the trained PyTorch CNN model and
prints predicted category, confidence, data provenance, and top probabilities.

Usage:
    python scripts/predict_audio.py data/sample/sample_siren_urban.wav
    python scripts/predict_audio.py path/to/my_audio.wav --model models/cnn_urban_sound.pth
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import settings
from src.inference.predict import SoundClassifier


def main():
    parser = argparse.ArgumentParser(description="Predict acoustic category from audio file.")
    parser.add_argument("file_path", type=str, nargs="?", default="data/sample/sample_siren_urban.wav",
                        help="Path to audio file (.wav)")
    parser.add_argument("--model", type=str, default=str(settings.DEFAULT_MODEL_PATH),
                        help="Path to trained model checkpoint (.pth)")
    parser.add_argument("--allow-untrained", action="store_true",
                        help="Allow fallback to untrained baseline weights or synthetic models for development testing")
    args = parser.parse_args()

    audio_path = Path(args.file_path)
    if not audio_path.exists():
        print(f"[ERROR] Audio file not found: {audio_path}")
        print("Tip: Run `python scripts/generate_sample_audio.py` to create a test sample.")
        sys.exit(1)

    print("=" * 60)
    print("ACOUSTIC EVENT CLASSIFICATION")
    print("=" * 60)
    print(f"  Input File:   {audio_path}")
    print(f"  Model Path:   {args.model}")
    print("-" * 60)

    try:
        classifier = SoundClassifier(model_path=args.model, allow_untrained=args.allow_untrained)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"[ERROR] Checkpoint verification/provenance error:\n        {e}")
        sys.exit(1)

    result = classifier.predict_file(audio_path)

    predicted_class = result["predicted_class"]
    confidence = result["confidence"]
    is_trained = result.get("is_trained", False)
    is_synthetic = result.get("is_synthetic", False)
    data_type = result.get("training_data_type", "unknown")

    if is_trained and not is_synthetic and data_type == "urbansound8k":
        status_str = "YES (Verified UrbanSound8K)"
    elif is_synthetic:
        status_str = "NO (Synthetic Development Model)"
    else:
        status_str = f"NO (Untrained / Baseline - {data_type})"

    print(f"  PREDICTED CLASS:  {predicted_class.upper()}")
    print(f"  CONFIDENCE:       {confidence * 100:.2f}%")
    print(f"  IS TRAINED MODEL: {status_str}")
    print(f"  DATA PROVENANCE:  {data_type.upper()}")
    print("-" * 60)
    print("  TOP PREDICTIONS:")

    # Sort probabilities descending
    sorted_probs = sorted(
        result["probabilities"].items(),
        key=lambda item: item[1],
        reverse=True,
    )

    for rank, (cls_name, prob) in enumerate(sorted_probs[:5], 1):
        bar_len = int(prob * 30)
        bar = "#" * bar_len + "-" * (30 - bar_len)
        print(f"    {rank}. {cls_name:<18} [{bar}] {prob * 100:5.1f}%")

    print("=" * 60)


if __name__ == "__main__":
    main()
