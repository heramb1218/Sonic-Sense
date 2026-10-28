"""
UrbanSound8K Download CLI Script using Soundata — Sonic-Sense

Downloads and extracts the official UrbanSound8K dataset from Zenodo
directly into the project's data/raw/UrbanSound8K/ directory.

Usage:
    python scripts/download_urbansound8k.py
    python scripts/download_urbansound8k.py --cleanup
"""

import sys
import os
import argparse
import logging
from pathlib import Path

# Explicit import required to ensure urllib.request is populated on urllib
# (prevents AttributeError in soundata.download_utils)
import urllib.request

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import settings

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Download official UrbanSound8K dataset using soundata.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(settings.URBANSOUND8K_DIR),
        help="Target destination directory for UrbanSound8K dataset",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Delete UrbanSound8K.tar.gz archive after successful extraction",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if files exist",
    )
    args = parser.parse_args()

    target_dir = Path(args.data_dir).resolve()
    print("=" * 70)
    print("URBANSOUND8K DATASET DOWNLOADER (via Soundata)")
    print("=" * 70)
    print(f"Target Directory: {target_dir}")
    print(f"Cleanup Archive:  {args.cleanup}")
    print("-" * 70)

    # Check for existing data conflict
    audio_dir = target_dir / "audio"
    metadata_file = target_dir / "metadata" / "UrbanSound8K.csv"
    if (audio_dir.exists() or metadata_file.exists()) and not args.force:
        wav_count = len(list(audio_dir.glob("fold*/*.wav"))) if audio_dir.exists() else 0
        if wav_count > 0:
            print(f"[CONFLICT] Destination already contains {wav_count} WAV files and/or metadata.")
            print("To re-download and overwrite, pass --force.")
            sys.exit(1)

    target_dir.mkdir(parents=True, exist_ok=True)

    import soundata

    logger.info("Initializing soundata 'urbansound8k' dataset...")
    dataset = soundata.initialize("urbansound8k", data_home=str(target_dir))

    logger.info(f"Starting download to {target_dir}...")
    print("Downloading dataset (~5.6 GB). This may take several minutes depending on network speed...")
    sys.stdout.flush()

    try:
        dataset.download(force_overwrite=args.force, cleanup=args.cleanup)
        print("\n[SUCCESS] Download and extraction completed successfully!")
    except Exception as exc:
        print(f"\n[ERROR] Download failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

