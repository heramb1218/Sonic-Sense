"""
Inference Engine — Predict sound category and confidence from raw audio.

Loads a trained UrbanSoundCNN checkpoint and classifies single audio
arrays or audio files with strict training-data provenance enforcement.

Usage:
    from src.inference.predict import SoundClassifier

    classifier = SoundClassifier("models/cnn_urban_sound.pth")
    result = classifier.predict_file("data/sample/sample_siren_urban.wav")
    print(result["predicted_class"], result["confidence"])
"""

import logging
from pathlib import Path
from typing import Optional, Union
import numpy as np
import torch
import torch.nn.functional as F

from src.config.settings import settings
from src.audio.processor import load_audio, preprocess_audio
from src.features.extractor import extract_log_mel_spectrogram
from src.models.cnn import UrbanSoundCNN

logger = logging.getLogger(__name__)


class SoundClassifier:
    """Acoustic classifier wrapping the trained PyTorch CNN model with provenance validation."""

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        device: Optional[torch.device] = None,
        allow_untrained: bool = False,
    ):
        """Initialize classifier with trained checkpoint.

        Args:
            model_path: Path to checkpoint .pth file. Defaults to settings.DEFAULT_MODEL_PATH.
            device: torch.device (defaults to GPU if available, else CPU).
            allow_untrained: If True, allows running with uninitialized baseline weights
                or synthetic/unverified development checkpoints. STRICTLY for development and testing.
                Defaults to False (enforces genuine UrbanSound8K checkpoint).

        Raises:
            FileNotFoundError: If checkpoint does not exist and allow_untrained=False.
            ValueError: If checkpoint is incompatible, synthetic, unverified, or corrupted
                when allow_untrained=False.
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = UrbanSoundCNN(num_classes=settings.NUM_CLASSES)
        self.allow_untrained = allow_untrained
        self.is_synthetic: bool = False
        self.training_data_type: str = "untrained"
        self.is_trained: bool = False

        target_path = Path(model_path) if model_path is not None else settings.DEFAULT_MODEL_PATH

        if target_path.exists():
            try:
                checkpoint = torch.load(str(target_path), map_location=self.device, weights_only=False)
            except Exception as e:
                raise ValueError(f"Failed to load checkpoint file at '{target_path}': {e}") from e

            # Validate checkpoint metadata if present
            if isinstance(checkpoint, dict):
                arch = checkpoint.get("architecture")
                if arch is not None and arch != "UrbanSoundCNN":
                    raise ValueError(
                        f"Incompatible architecture in checkpoint: '{arch}'. Expected 'UrbanSoundCNN'."
                    )

                num_cls = checkpoint.get("num_classes")
                if num_cls is not None and num_cls != settings.NUM_CLASSES:
                    raise ValueError(
                        f"Incompatible num_classes in checkpoint: {num_cls}. Expected {settings.NUM_CLASSES}."
                    )

                # Training-data provenance validation (Task 3)
                data_type = checkpoint.get("training_data_type")
                is_synth = checkpoint.get("is_synthetic")

                if is_synth is True or data_type == "synthetic":
                    if not allow_untrained:
                        raise ValueError(
                            f"Model checkpoint at '{target_path}' was trained on synthetic data "
                            f"(training_data_type='synthetic', is_synthetic=True) and cannot be used for "
                            f"genuine acoustic classification. To permit synthetic/development inference, "
                            f"pass allow_untrained=True."
                        )
                    self.is_synthetic = True
                    self.training_data_type = "synthetic"
                    self.is_trained = False
                    logger.warning(
                        f"Loaded synthetic development checkpoint from {target_path} (allow_untrained=True). "
                        "Predictions are marked as synthetic and unverified."
                    )
                elif data_type != "urbansound8k" or is_synth is not False:
                    # Missing, ambiguous, or legacy provenance
                    if not allow_untrained:
                        raise ValueError(
                            f"Model checkpoint at '{target_path}' lacks verified UrbanSound8K training data provenance "
                            f"(found training_data_type={data_type!r}, is_synthetic={is_synth!r}). "
                            f"Genuine inference requires a checkpoint trained on UrbanSound8K. "
                            f"For development/testing, pass allow_untrained=True."
                        )
                    self.is_synthetic = bool(is_synth) if is_synth is not None else False
                    self.training_data_type = str(data_type or "unverified_legacy")
                    self.is_trained = False
                    logger.warning(
                        f"Loaded unverified checkpoint from {target_path} (allow_untrained=True). "
                        f"Provenance: {self.training_data_type}."
                    )
                else:
                    # Genuine verified UrbanSound8K checkpoint
                    self.is_synthetic = False
                    self.training_data_type = "urbansound8k"
                    self.is_trained = True
                    logger.info(f"Loaded verified UrbanSound8K trained model from {target_path}")

                state_dict = checkpoint.get("model_state_dict", checkpoint)
                self.classes = checkpoint.get("classes", settings.URBAN_SOUND_CLASSES)
            else:
                # Raw state dict without metadata
                if not allow_untrained:
                    raise ValueError(
                        f"Model checkpoint at '{target_path}' is a raw state dict without training data "
                        f"provenance metadata. Genuine inference requires a checkpoint trained on "
                        f"UrbanSound8K with verified metadata. Pass allow_untrained=True for development."
                    )
                state_dict = checkpoint
                self.classes = settings.URBAN_SOUND_CLASSES
                self.is_synthetic = False
                self.training_data_type = "unverified_raw"
                self.is_trained = False
                logger.warning(f"Loaded raw unverified state dict from {target_path} (allow_untrained=True).")

            try:
                self.model.load_state_dict(state_dict)
            except Exception as e:
                raise ValueError(
                    f"Model state dict at '{target_path}' could not be loaded into UrbanSoundCNN: {e}"
                ) from e
        else:
            if not allow_untrained:
                raise FileNotFoundError(
                    f"Model checkpoint not found at: {target_path.resolve()}\n"
                    f"To train the CNN model on UrbanSound8K, run:\n"
                    f"    python scripts/train_cnn.py --epochs 20 --augment\n"
                    f"For development/testing with baseline weights, pass allow_untrained=True."
                )
            logger.warning(
                f"Model checkpoint not found at {target_path}. "
                "Running with uninitialized baseline weights (allow_untrained=True)."
            )
            self.classes = settings.URBAN_SOUND_CLASSES
            self.is_synthetic = False
            self.training_data_type = "untrained"
            self.is_trained = False

        self.model.to(self.device)
        self.model.eval()

    def predict_audio(self, audio: np.ndarray, sr: int) -> dict:
        """Classify a raw audio waveform array.

        Args:
            audio: 1D numpy array of audio samples.
            sr: Sample rate in Hz.

        Returns:
            Dictionary with:
                - predicted_class (str): Name of predicted sound category
                - class_id (int): Index of predicted category (0-9)
                - confidence (float): Probability score of top class [0.0 - 1.0]
                - probabilities (dict[str, float]): Probability for every class
                - is_trained (bool): Whether predictions are from a verified trained checkpoint
                - is_synthetic (bool): Whether predictions are from synthetic training
                - training_data_type (str): Provenance of the model ("urbansound8k", "synthetic", etc.)
        """
        # Step 1: Preprocess (normalize)
        norm_audio = preprocess_audio(audio, sr)

        # Step 2: Extract Log-Mel Spectrogram (128, 173)
        log_mel = extract_log_mel_spectrogram(norm_audio, sr)

        # Step 3: Create batch tensor (1, 1, 128, 173)
        tensor = torch.from_numpy(log_mel).unsqueeze(0).unsqueeze(0).float().to(self.device)

        # Step 4: Model forward pass with softmax
        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        class_id = int(np.argmax(probs))
        confidence = float(probs[class_id])
        predicted_class = self.classes.get(class_id, "unknown")

        class_probs = {
            self.classes.get(i, f"class_{i}"): float(prob)
            for i, prob in enumerate(probs)
        }

        return {
            "predicted_class": predicted_class,
            "class_id": class_id,
            "confidence": round(confidence, 4),
            "probabilities": class_probs,
            "is_trained": self.is_trained,
            "is_synthetic": self.is_synthetic,
            "training_data_type": self.training_data_type,
        }

    def predict_file(self, file_path: Union[str, Path]) -> dict:
        """Load audio file and predict its acoustic category.

        Args:
            file_path: Path to .wav or other audio file.

        Returns:
            Prediction result dict including file metadata.
        """
        file_path = Path(file_path)
        audio, sr = load_audio(file_path)
        result = self.predict_audio(audio, sr)
        result["file_path"] = str(file_path)
        return result
