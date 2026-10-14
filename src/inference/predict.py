"""
Inference Engine — Predict sound category and confidence from raw audio.

Loads a trained UrbanSoundCNN checkpoint and classifies single audio
arrays or audio files.

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
    """Acoustic classifier wrapping the trained PyTorch CNN model."""

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        device: Optional[torch.device] = None,
    ):
        """Initialize classifier with trained checkpoint.

        Args:
            model_path: Path to checkpoint .pth file. If None or missing,
                model is initialized with random weights for demo/testing.
            device: torch.device (defaults to GPU if available, else CPU).
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = UrbanSoundCNN(num_classes=settings.NUM_CLASSES)

        model_path = Path(model_path or settings.DEFAULT_MODEL_PATH)
        if model_path.exists():
            checkpoint = torch.load(str(model_path), map_location=self.device)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            self.model.load_state_dict(state_dict)
            self.classes = checkpoint.get("classes", settings.URBAN_SOUND_CLASSES)
            logger.info(f"Loaded model weights from {model_path}")
        else:
            logger.warning(
                f"Model checkpoint not found at {model_path}. "
                "Running with uninitialized baseline weights."
            )
            self.classes = settings.URBAN_SOUND_CLASSES

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
        result["filename"] = file_path.name
        return result
