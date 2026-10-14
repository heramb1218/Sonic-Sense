"""
Simulated Acoustic Sensor — Smartphone & IoT Sensor Agent

Simulates an acoustic sensing node (such as a crowdsourcing smartphone or
stationary IoT monitor) moving through an urban environment.

Key responsibilities:
- Mobility and simulated GPS trajectory with realistic GPS accuracy jitter
- Microphone decibel estimation (simulated SPL dB)
- Hardware calibration offset simulation (device heterogeneity)
- ML model inference for sound classification
- Emitting standardized JSON noise observation events (Section 16)
"""

import math
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union
import numpy as np

from src.config.settings import settings
from src.simulation.zones import UrbanZone, haversine_distance
from src.audio.processor import load_audio
from src.inference.predict import SoundClassifier
from src.spatial.h3_grid import H3GridManager


# Representative smartphone models and typical acoustic calibration offsets
DEVICE_MODELS_POOL = [
    {"model": "Apple iPhone 14 Pro", "offset": 1.2, "gps_std": 3.5},
    {"model": "Google Pixel 7", "offset": -1.8, "gps_std": 4.0},
    {"model": "Samsung Galaxy S23", "offset": 0.5, "gps_std": 4.5},
    {"model": "OnePlus 11", "offset": -2.3, "gps_std": 5.0},
    {"model": "Xiaomi 13 Pro", "offset": 0.9, "gps_std": 5.5},
    {"model": "AcousticStation-IoT-Fixed", "offset": 0.0, "gps_std": 1.5},
]


def estimate_db_from_audio(audio: np.ndarray, zone: Optional[UrbanZone] = None) -> float:
    """Estimate equivalent sound pressure level (dB SPL) from audio array.

    NOTE ON SCIENTIFIC INTEGRITY:
    This value is an acoustic estimate calculated from normalized digital RMS
    energy scaled to representative urban SPL values. It is clearly labeled
    as estimated/simulated decibels, not a certified Class 1 Sound Level Meter reading.

    Args:
        audio: 1D numpy array of audio samples.
        zone: Optional UrbanZone supplying environmental ambient baseline.

    Returns:
        Estimated noise level in dB (float, rounded to 1 decimal place).
    """
    rms = np.sqrt(np.mean(audio.astype(np.float64) ** 2))
    # 20 * log10(rms) maps to [-inf, 0]. Add reference offset ~92 dB so normal speech is ~60dB, loud ~85dB
    if rms < 1e-6:
        calculated_db = 35.0
    else:
        calculated_db = 20.0 * np.log10(rms) + 94.0

    # If zone is provided, blend slightly with zone baseline for realistic environmental distribution
    if zone is not None:
        zone_sample = np.random.normal(zone.baseline_db_mean, zone.baseline_db_std)
        final_db = 0.7 * calculated_db + 0.3 * zone_sample
    else:
        final_db = calculated_db

    # Clip to realistic environmental limits [30.0 dB, 115.0 dB]
    final_db = float(np.clip(final_db, 30.0, 115.0))
    return round(final_db, 1)


class SimulatedSensor:
    """Autonomous simulated mobile/stationary acoustic sensor."""

    def __init__(
        self,
        device_id: str,
        device_model: Optional[str] = None,
        calibration_offset: Optional[float] = None,
        initial_lat: Optional[float] = None,
        initial_lon: Optional[float] = None,
        current_zone: Optional[UrbanZone] = None,
        speed_mps: float = 1.4,  # Average pedestrian walking speed ~1.4 m/s (5 km/h)
    ):
        self.device_id = device_id

        # Device hardware configuration
        meta = DEVICE_MODELS_POOL[hash(device_id) % len(DEVICE_MODELS_POOL)]
        self.device_model = device_model or meta["model"]
        self.calibration_offset = calibration_offset if calibration_offset is not None else meta["offset"]
        self.base_gps_std = meta["gps_std"]

        # Coordinates & Mobility
        self.current_zone = current_zone
        self.lat = initial_lat if initial_lat is not None else settings.CITY_LAT
        self.lon = initial_lon if initial_lon is not None else settings.CITY_LON
        self.speed_mps = speed_mps
        self.heading = float(np.random.uniform(0, 360))  # Direction in degrees

    def step(self, dt_seconds: float = 1.0) -> None:
        """Advance sensor position based on movement speed and heading with boundary reflection."""
        # Gentle heading perturbation (random walk)
        self.heading += float(np.random.normal(0, 15))
        self.heading %= 360.0

        dist = self.speed_mps * dt_seconds
        rad = math.radians(self.heading)

        # 1 deg lat ≈ 111,111m; 1 deg lon ≈ 111,111m * cos(lat)
        delta_lat = (dist * math.cos(rad)) / 111111.0
        delta_lon = (dist * math.sin(rad)) / (111111.0 * math.cos(math.radians(self.lat)))

        new_lat = self.lat + delta_lat
        new_lon = self.lon + delta_lon

        # If anchored to a zone, ensure the sensor stays generally within zone perimeter
        if self.current_zone is not None:
            if not self.current_zone.contains(new_lat, new_lon):
                # Reverse heading towards zone center
                center_heading = math.degrees(math.atan2(
                    self.current_zone.center_lon - self.lon,
                    self.current_zone.center_lat - self.lat,
                )) % 360.0
                self.heading = center_heading
                return

        self.lat = round(new_lat, 6)
        self.lon = round(new_lon, 6)

    def get_gps_accuracy(self) -> float:
        """Simulate realistic instantaneous GPS accuracy in meters."""
        jitter = np.random.exponential(1.2)
        return round(float(self.base_gps_std + jitter), 1)

    def capture_observation(
        self,
        audio_file_path: Optional[Union[str, Path]] = None,
        audio_array: Optional[np.ndarray] = None,
        classifier: Optional[SoundClassifier] = None,
    ) -> dict:
        """Capture an acoustic measurement and construct the standard event record.

        Follows schema defined in Master Prompt Section 16.
        """
        sr = settings.SAMPLE_RATE
        duration = settings.AUDIO_DURATION

        # 1. Acquire audio
        if audio_array is not None:
            audio = audio_array
            source_file = str(audio_file_path) if audio_file_path else "synthetic_stream.wav"
        elif audio_file_path is not None and Path(audio_file_path).exists():
            audio, sr = load_audio(audio_file_path)
            source_file = str(Path(audio_file_path).name)
        else:
            # Fallback synthetic tone
            t = np.linspace(0, duration, int(sr * duration), endpoint=False)
            audio = (0.5 * np.sin(2 * np.pi * 500 * t) + 0.05 * np.random.randn(len(t))).astype(np.float32)
            source_file = "synthetic_default.wav"

        # 2. Estimate decibels & apply calibration offset
        raw_db = estimate_db_from_audio(audio, zone=self.current_zone)
        calibrated_db = round(raw_db + self.calibration_offset, 1)

        # 3. ML sound classification
        if classifier is not None:
            pred = classifier.predict_audio(audio, sr)
            sound_class = pred["predicted_class"]
            confidence = pred["confidence"]
        else:
            # Default placeholder when classifier is omitted
            sound_class = "ambient"
            confidence = 0.50

        # 4. Construct standard data event (Section 16)
        event = {
            "event_id": str(uuid.uuid4()),
            "device_id": self.device_id,
            "device_model": self.device_model,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "latitude": self.lat,
            "longitude": self.lon,
            "gps_accuracy": self.get_gps_accuracy(),
            "audio_file": source_file,
            "sample_rate": sr,
            "duration": duration,
            "noise_level_db": raw_db,
            "sound_class": sound_class,
            "confidence": confidence,
            "calibration_offset": round(self.calibration_offset, 2),
            "calibrated_db": calibrated_db,
            "h3_cell": H3GridManager.lat_lng_to_cell(self.lat, self.lon),
            "anomaly_score": None,   # Computed in Phase 9
        }

        return event
