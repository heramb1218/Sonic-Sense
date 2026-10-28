"""
Simulation Fleet Manager — Orchestrates heterogeneous sensor fleet and audio stream

Coordinates:
- Sensor fleet distributed across urban zones
- Audio sample pool (from UrbanSound8K dataset, data/sample/, or synthetic audio generator)
- Continuous observation event generation with configurable interval and callback hooks

Usage:
    from src.simulation.manager import SimulationFleet

    fleet = SimulationFleet(num_sensors=5)
    event = fleet.generate_next_event()
"""

import time
import logging
from pathlib import Path
from typing import Optional, Callable, Sequence
import numpy as np

from src.config.settings import settings
from src.simulation.zones import ZoneRegistry, UrbanZone
from src.simulation.sensors import SimulatedSensor
from src.inference.predict import SoundClassifier

logger = logging.getLogger(__name__)


class SimulationFleet:
    """Manages a network of simulated urban acoustic sensors."""

    def __init__(
        self,
        num_sensors: int = 5,
        classifier: Optional[SoundClassifier] = None,
        model_path: Optional[str] = None,
    ):
        self.num_sensors = num_sensors
        self.zone_registry = ZoneRegistry()
        self.zones = self.zone_registry.get_all_zones()

        # Shared ML inference model
        if classifier is not None:
            self.classifier = classifier
        else:
            try:
                self.classifier = SoundClassifier(model_path=model_path or settings.DEFAULT_MODEL_PATH)
            except (FileNotFoundError, ValueError) as exc:
                logger.info(
                    "Simulation fleet running with untrained baseline classifier (allow_untrained=True): %s",
                    exc,
                )
                self.classifier = SoundClassifier(model_path=model_path, allow_untrained=True)

        # Initialize sensors distributed across zones
        self.sensors = self._init_sensors()
        self.audio_pool = self._build_audio_pool()
        self._sensor_idx = 0

    def _init_sensors(self) -> list[SimulatedSensor]:
        sensors = []
        for i in range(self.num_sensors):
            device_id = f"sensor_{i+1:03d}"
            assigned_zone = self.zones[i % len(self.zones)]
            init_lat, init_lon = self.zone_registry.sample_coordinate_in_zone(assigned_zone)

            sensor = SimulatedSensor(
                device_id=device_id,
                initial_lat=init_lat,
                initial_lon=init_lon,
                current_zone=assigned_zone,
                speed_mps=float(np.random.uniform(1.0, 3.5)),  # Pedestrian to light cycling
            )
            sensors.append(sensor)
        return sensors

    def _build_audio_pool(self) -> list[Path]:
        """Collect available audio files from UrbanSound8K or sample folders."""
        pool = []

        # Check sample dir first
        sample_dir = settings.DATA_DIR / "sample"
        if sample_dir.exists():
            pool.extend(list(sample_dir.glob("*.wav")))

        # Check UrbanSound8K dataset audio folds
        audio_dir = settings.URBANSOUND8K_AUDIO_DIR
        if audio_dir.exists():
            for fold in range(1, 11):
                fold_dir = audio_dir / f"fold{fold}"
                if fold_dir.exists():
                    pool.extend(list(fold_dir.glob("*.wav"))[:50])

        return pool

    def _synthesize_zone_audio(self, zone: UrbanZone) -> tuple[np.ndarray, str]:
        """Generate a synthetic audio waveform characteristic of the urban zone."""
        sr = settings.SAMPLE_RATE
        duration = settings.AUDIO_DURATION
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)

        # Choose signature sound based on zone type
        if zone.zone_type == "traffic":
            # Low engine rumble + occasional horn
            engine = 0.4 * np.sin(2 * np.pi * 90 * t) + 0.2 * np.sin(2 * np.pi * 180 * t)
            horn = 0.5 * np.sin(2 * np.pi * 420 * t) * (t > 1.5) * (t < 2.2)
            noise = 0.1 * np.random.randn(len(t))
            audio = (engine + horn + noise).astype(np.float32)
            name = "traffic_simulation.wav"

        elif zone.zone_type == "construction":
            # Repetitive high-energy impact / drilling
            pulse = np.sin(2 * np.pi * 12 * t) > 0.3
            drill = 0.6 * np.sin(2 * np.pi * 650 * t) * pulse
            noise = 0.2 * np.random.randn(len(t))
            audio = (drill + noise).astype(np.float32)
            name = "construction_simulation.wav"

        elif zone.zone_type == "quiet":
            # Soft ambient wind / distant tone
            ambient = 0.08 * np.sin(2 * np.pi * 220 * t) + 0.04 * np.random.randn(len(t))
            audio = ambient.astype(np.float32)
            name = "quiet_park_ambient.wav"

        else:
            # Residential / Commercial: modulated tones
            voice = 0.3 * np.sin(2 * np.pi * 320 * t) + 0.15 * np.sin(2 * np.pi * 580 * t)
            noise = 0.06 * np.random.randn(len(t))
            audio = (voice + noise).astype(np.float32)
            name = "urban_general.wav"

        # Peak normalize
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak

        return audio, name

    def generate_next_event(self) -> dict:
        """Advance one sensor, process an audio sample, and emit a standardized event."""
        sensor = self.sensors[self._sensor_idx % len(self.sensors)]
        self._sensor_idx += 1

        # Advance sensor mobility
        sensor.step(dt_seconds=settings.SIMULATION_INTERVAL)

        # Obtain audio: use disk file if available, else synthesize zone-specific audio
        if self.audio_pool and np.random.rand() < 0.6:
            chosen_file = self.audio_pool[np.random.randint(0, len(self.audio_pool))]
            event = sensor.capture_observation(
                audio_file_path=chosen_file,
                classifier=self.classifier,
            )
        else:
            audio_array, audio_name = self._synthesize_zone_audio(sensor.current_zone)
            event = sensor.capture_observation(
                audio_file_path=audio_name,
                audio_array=audio_array,
                classifier=self.classifier,
            )

        return event

    def run_simulation(
        self,
        num_events: int = 10,
        interval_sec: float = 1.0,
        callback: Optional[Callable[[dict], None]] = None,
    ) -> list[dict]:
        """Run a simulation loop generating events with a delay between emissions."""
        events = []
        for i in range(num_events):
            event = self.generate_next_event()
            events.append(event)

            if callback is not None:
                callback(event)

            if i < num_events - 1 and interval_sec > 0:
                time.sleep(interval_sec)

        return events
