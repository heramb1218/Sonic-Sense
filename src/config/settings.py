"""
Urban Noise Intelligence - Central Configuration

Reads settings from .env file and provides typed defaults.
All configurable parameters for the project are defined here.

Usage:
    from src.config.settings import Settings
    settings = Settings()
    sr = settings.SAMPLE_RATE
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
# Walk up from this file: src/config/settings.py -> src/config -> src -> project_root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")


class Settings:
    """Central configuration for the Urban Noise Intelligence platform.

    All parameters are read from environment variables with sensible defaults.
    Copy .env.example to .env to override any value.
    """

    def __init__(self):
        # ============================================
        # Audio Processing Parameters
        # ============================================

        # Sampling rate in Hz. UrbanSound8K is natively 22050 Hz.
        self.SAMPLE_RATE: int = int(os.getenv("SAMPLE_RATE", "22050"))

        # Fixed duration for all audio clips (seconds).
        # UrbanSound8K clips are <= 4 seconds.
        # Shorter clips are zero-padded, longer clips are truncated.
        self.AUDIO_DURATION: float = float(os.getenv("AUDIO_DURATION", "4.0"))

        # FFT window size. 2048 samples ≈ 93ms at 22050 Hz.
        # Controls frequency resolution of spectrograms.
        self.N_FFT: int = int(os.getenv("N_FFT", "2048"))

        # Hop length between successive FFT windows.
        # 512 samples ≈ 23ms at 22050 Hz.
        self.HOP_LENGTH: int = int(os.getenv("HOP_LENGTH", "512"))

        # Number of Mel filter banks.
        # 128 is standard for environmental sound classification.
        self.N_MELS: int = int(os.getenv("N_MELS", "128"))

        # ============================================
        # Directory Paths
        # ============================================
        self.DATA_DIR: Path = PROJECT_ROOT / os.getenv("DATA_DIR", "data")
        self.RAW_DATA_DIR: Path = PROJECT_ROOT / os.getenv("RAW_DATA_DIR", "data/raw")
        self.PROCESSED_DATA_DIR: Path = PROJECT_ROOT / os.getenv("PROCESSED_DATA_DIR", "data/processed")
        self.METADATA_DIR: Path = PROJECT_ROOT / os.getenv("METADATA_DIR", "data/metadata")
        self.URBANSOUND8K_DIR: Path = PROJECT_ROOT / os.getenv("URBANSOUND8K_DIR", "data/raw/UrbanSound8K")
        self.MODEL_DIR: Path = PROJECT_ROOT / os.getenv("MODEL_DIR", "models")
        self.DEFAULT_MODEL_PATH: Path = PROJECT_ROOT / os.getenv("MODEL_PATH", "models/cnn_urban_sound.pth")
        self.RESULTS_DIR: Path = PROJECT_ROOT / "results"

        # ============================================
        # Training Parameters
        # ============================================
        self.BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "32"))
        self.LEARNING_RATE: float = float(os.getenv("LEARNING_RATE", "0.001"))
        self.NUM_EPOCHS: int = int(os.getenv("NUM_EPOCHS", "20"))

        # ============================================
        # Simulation Parameters (Phase 4)
        # ============================================
        self.CITY_LAT: float = float(os.getenv("CITY_LAT", "18.5204"))
        self.CITY_LON: float = float(os.getenv("CITY_LON", "73.8567"))
        self.NUM_SENSORS: int = int(os.getenv("NUM_SENSORS", "5"))
        self.SIMULATION_INTERVAL: float = float(os.getenv("SIMULATION_INTERVAL", "2.0"))

        # ============================================
        # Streaming Parameters (Phase 5)
        # ============================================
        self.KAFKA_BROKER: str = os.getenv("KAFKA_BROKER", "localhost:9092")
        self.KAFKA_TOPIC_EVENTS: str = os.getenv("KAFKA_TOPIC_EVENTS", "urban-noise-events")
        self.KAFKA_TOPIC_PREDICTIONS: str = os.getenv("KAFKA_TOPIC_PREDICTIONS", "urban-noise-predictions")
        self.KAFKA_TOPIC_ANOMALIES: str = os.getenv("KAFKA_TOPIC_ANOMALIES", "urban-noise-anomalies")
        self.KAFKA_CONSUMER_GROUP: str = os.getenv("KAFKA_CONSUMER_GROUP", "urban-noise-processor-group")
        self.STREAMING_MODE: str = os.getenv("STREAMING_MODE", "auto")

        # ============================================
        # Database Parameters (Phase 6)
        # ============================================
        self.DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///urban_noise.db")
        self.POSTGRES_DATABASE_URL: str = os.getenv(
            "POSTGRES_DATABASE_URL",
            "postgresql://urban_admin:urban_secure_password_123@localhost:5432/urban_noise",
        )
        self.DB_ECHO: bool = os.getenv("DB_ECHO", "False").lower() in ("true", "1")

        # ============================================
        # Backend Parameters (Phase 7)
        # ============================================
        self.BACKEND_HOST: str = os.getenv("BACKEND_HOST", "127.0.0.1")
        self.BACKEND_PORT: int = int(os.getenv("BACKEND_PORT", "8000"))
        self.CORS_ORIGINS: list[str] = [
            origin.strip() for origin in os.getenv("CORS_ORIGINS", "*").split(",")
        ]

        # ============================================
        # Derived Paths
        # ============================================
        self.URBANSOUND8K_AUDIO_DIR: Path = self.URBANSOUND8K_DIR / "audio"
        self.URBANSOUND8K_METADATA: Path = self.URBANSOUND8K_DIR / "metadata" / "UrbanSound8K.csv"

        # ============================================
        # UrbanSound8K Class Labels
        # ============================================
        self.URBAN_SOUND_CLASSES: dict[int, str] = {
            0: "air_conditioner",
            1: "car_horn",
            2: "children_playing",
            3: "dog_bark",
            4: "drilling",
            5: "engine_idling",
            6: "gun_shot",
            7: "jackhammer",
            8: "siren",
            9: "street_music",
        }
        self.NUM_CLASSES: int = len(self.URBAN_SOUND_CLASSES)

    def get_fold_dir(self, fold: int) -> Path:
        """Get the directory path for a specific UrbanSound8K fold.

        Args:
            fold: Fold number (1-10).

        Returns:
            Path to the fold directory.
        """
        if not 1 <= fold <= 10:
            raise ValueError(f"Fold must be between 1 and 10, got {fold}")
        return self.URBANSOUND8K_AUDIO_DIR / f"fold{fold}"

    def __repr__(self) -> str:
        return (
            f"Settings(\n"
            f"  SAMPLE_RATE={self.SAMPLE_RATE},\n"
            f"  AUDIO_DURATION={self.AUDIO_DURATION},\n"
            f"  N_FFT={self.N_FFT},\n"
            f"  HOP_LENGTH={self.HOP_LENGTH},\n"
            f"  N_MELS={self.N_MELS},\n"
            f"  URBANSOUND8K_DIR={self.URBANSOUND8K_DIR},\n"
            f"  NUM_CLASSES={self.NUM_CLASSES}\n"
            f")"
        )


# Singleton instance for convenience
settings = Settings()
