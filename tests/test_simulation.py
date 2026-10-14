"""
Unit Tests for Phase 4 — Sensor Simulation and Urban Mobility

Tests verify:
- Geographic zone registration, bounds, and Haversine distance calculations
- Sensor movement updates and boundary preservation
- Microphone noise level (dB) estimation from audio energy
- Hardware calibration offset formula (calibrated_db = noise_level_db + offset)
- Strict compliance with Section 16 standard event schema
- Fleet orchestration and observation streaming

Usage:
    pytest tests/test_simulation.py -v
"""

import sys
import numpy as np
import pytest

# Add project root to path
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import settings
from src.simulation.zones import ZoneRegistry, haversine_distance
from src.simulation.sensors import SimulatedSensor, estimate_db_from_audio
from src.simulation.manager import SimulationFleet


class TestUrbanZones:
    """Tests for geographic zone logic."""

    def test_zone_registry_initialization(self):
        registry = ZoneRegistry()
        zones = registry.get_all_zones()
        assert len(zones) == 5
        zone_types = {z.zone_type for z in zones}
        assert {"residential", "traffic", "commercial", "construction", "quiet"} <= zone_types

    def test_haversine_distance_calculation(self):
        # Distance from (0, 0) to (0, 1) degree longitude at equator is ~111.3 km
        dist = haversine_distance(0.0, 0.0, 0.0, 1.0)
        assert 110_000 < dist < 112_000

    def test_find_zone_for_center_point(self):
        registry = ZoneRegistry()
        zone = registry.zones[0]
        found = registry.find_zone_for_point(zone.center_lat, zone.center_lon)
        assert found is not None
        assert found.zone_id == zone.zone_id


class TestSensorMobilityAndAcoustics:
    """Tests for simulated sensor physics and noise calculations."""

    def test_sensor_step_updates_coordinates(self):
        sensor = SimulatedSensor("test_sensor_1", initial_lat=18.5204, initial_lon=73.8567, speed_mps=5.0)
        initial_lat, initial_lon = sensor.lat, sensor.lon
        sensor.step(dt_seconds=2.0)
        # Position should have shifted
        assert (sensor.lat, sensor.lon) != (initial_lat, initial_lon)
        # Shift should be small (within reasonable walking/cycling distance)
        assert abs(sensor.lat - initial_lat) < 0.005
        assert abs(sensor.lon - initial_lon) < 0.005

    def test_noise_level_estimation_range(self):
        # Loud signal
        loud_audio = 0.9 * np.ones(22050, dtype=np.float32)
        loud_db = estimate_db_from_audio(loud_audio)

        # Quiet signal
        quiet_audio = 0.001 * np.ones(22050, dtype=np.float32)
        quiet_db = estimate_db_from_audio(quiet_audio)

        assert loud_db > quiet_db
        assert 30.0 <= quiet_db <= 115.0
        assert 30.0 <= loud_db <= 115.0

    def test_calibration_offset_application(self):
        sensor = SimulatedSensor(
            device_id="test_sensor_cal",
            calibration_offset=-2.5,
        )
        audio = 0.1 * np.ones(1000, dtype=np.float32)
        event = sensor.capture_observation(audio_array=audio)

        expected_calibrated = round(event["noise_level_db"] - 2.5, 1)
        assert event["calibrated_db"] == expected_calibrated
        assert event["calibration_offset"] == -2.5


class TestStandardEventSchema:
    """Tests confirming strict adherence to Master Prompt Section 16 schema."""

    def test_event_schema_keys(self):
        sensor = SimulatedSensor("test_sensor_schema")
        event = sensor.capture_observation()

        required_keys = {
            "event_id",
            "device_id",
            "device_model",
            "timestamp",
            "latitude",
            "longitude",
            "gps_accuracy",
            "audio_file",
            "sample_rate",
            "duration",
            "noise_level_db",
            "sound_class",
            "confidence",
            "calibration_offset",
            "calibrated_db",
            "h3_cell",
            "anomaly_score",
        }

        assert required_keys.issubset(set(event.keys())), f"Missing keys: {required_keys - set(event.keys())}"
        assert isinstance(event["event_id"], str) and len(event["event_id"]) > 10
        assert isinstance(event["latitude"], float)
        assert isinstance(event["longitude"], float)
        assert isinstance(event["gps_accuracy"], float)
        assert 0.0 <= event["confidence"] <= 1.0


class TestSimulationFleet:
    """Tests for fleet orchestration."""

    def test_fleet_generates_valid_event_stream(self):
        fleet = SimulationFleet(num_sensors=3)
        events = fleet.run_simulation(num_events=5, interval_sec=0.0)

        assert len(events) == 5
        device_ids = {e["device_id"] for e in events}
        # Events should be emitted across multiple sensors
        assert len(device_ids) >= 2
        for ev in events:
            assert ev["sound_class"] in settings.URBAN_SOUND_CLASSES.values() or ev["sound_class"] == "ambient"
