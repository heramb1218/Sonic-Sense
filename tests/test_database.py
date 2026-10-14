"""
Unit Tests for Phase 6 — Spatial-Temporal Database and Queries (Section 20)

Tests verify:
- Table and index provisioning on in-memory SQLite database
- Single record and transactional batch insertions
- Query 1: Latest observations retrieval
- Query 2: Time-range window filtering
- Query 3: Geographic bounding box viewport queries
- Query 4: Radial Haversine distance search
- Query 5: Sound class filtering
- Query 6: Spatial area noise aggregations (avg, min, max, count)
- Query 7: Class frequency distributions
- Query 8: Historical baseline calculation (mean, standard deviation)

Usage:
    pytest tests/test_database.py -v
"""

import sys
from datetime import datetime, timezone, timedelta
import pytest

# Add project root to path
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.database.connection import DatabaseManager
from src.database.repository import ObservationRepository


@pytest.fixture
def repo():
    """Create a fresh in-memory database and repository for each test."""
    db_mgr = DatabaseManager(db_url="sqlite:///:memory:")
    return ObservationRepository(db_manager=db_mgr)


@pytest.fixture
def sample_event():
    return {
        "event_id": "test_ev_001",
        "device_id": "sensor_001",
        "device_model": "Google Pixel 7",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "latitude": 18.5204,
        "longitude": 73.8567,
        "gps_accuracy": 4.5,
        "audio_file": "test.wav",
        "noise_level_db": 74.0,
        "sound_class": "siren",
        "confidence": 0.92,
        "calibration_offset": -1.8,
        "calibrated_db": 72.2,
        "h3_cell": "8860145b53fffff",
        "anomaly_score": None,
    }


class TestDatabaseRepository:
    """Tests for spatial-temporal CRUD and queries."""

    def test_insert_and_get_latest(self, repo, sample_event):
        obs = repo.insert_observation(sample_event)
        assert obs.id is not None
        assert obs.event_id == "test_ev_001"

        latest = repo.get_latest_observations(limit=5)
        assert len(latest) == 1
        assert latest[0].event_id == "test_ev_001"
        assert latest[0].calibrated_db == 72.2

    def test_insert_batch(self, repo):
        events = [
            {
                "event_id": f"batch_{i}",
                "device_id": f"sensor_{i}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "latitude": 18.52 + i * 0.001,
                "longitude": 73.85 + i * 0.001,
                "noise_level_db": 70.0 + i,
                "sound_class": "traffic" if i % 2 == 0 else "siren",
                "confidence": 0.85,
                "calibration_offset": 0.0,
                "calibrated_db": 70.0 + i,
            }
            for i in range(5)
        ]
        count = repo.insert_batch(events)
        assert count == 5
        assert len(repo.get_latest_observations(limit=10)) == 5

    def test_time_range_query(self, repo):
        now = datetime.now(timezone.utc)
        # 1 event 1 hour ago, 1 event 5 hours ago, 1 event 12 hours ago
        repo.insert_observation({
            "event_id": "time_1h",
            "device_id": "s1",
            "timestamp": (now - timedelta(hours=1)).isoformat(),
            "latitude": 18.52,
            "longitude": 73.85,
            "noise_level_db": 65.0,
            "sound_class": "car_horn",
            "confidence": 0.8,
            "calibrated_db": 65.0,
        })
        repo.insert_observation({
            "event_id": "time_5h",
            "device_id": "s1",
            "timestamp": (now - timedelta(hours=5)).isoformat(),
            "latitude": 18.52,
            "longitude": 73.85,
            "noise_level_db": 75.0,
            "sound_class": "car_horn",
            "confidence": 0.8,
            "calibrated_db": 75.0,
        })

        # Query range between 3h ago and now -> should only find time_1h
        results = repo.get_observations_in_time_range(
            start_time=now - timedelta(hours=3),
            end_time=now,
        )
        assert len(results) == 1
        assert results[0].event_id == "time_1h"

    def test_spatial_bounding_box_query(self, repo):
        # Point inside Pune center
        repo.insert_observation({
            "event_id": "pune_inside",
            "device_id": "s1",
            "latitude": 18.5204,
            "longitude": 73.8567,
            "noise_level_db": 60.0,
            "sound_class": "dog_bark",
            "confidence": 0.9,
            "calibrated_db": 60.0,
        })
        # Point far away in Mumbai
        repo.insert_observation({
            "event_id": "mumbai_outside",
            "device_id": "s2",
            "latitude": 19.0760,
            "longitude": 72.8777,
            "noise_level_db": 80.0,
            "sound_class": "siren",
            "confidence": 0.9,
            "calibrated_db": 80.0,
        })

        # Query bounding box around Pune only
        results = repo.get_observations_in_bounding_box(
            min_lat=18.50, min_lon=73.80, max_lat=18.55, max_lon=73.90
        )
        assert len(results) == 1
        assert results[0].event_id == "pune_inside"

    def test_radial_search_query(self, repo):
        center_lat, center_lon = 18.5204, 73.8567
        # Point ~300 meters away
        repo.insert_observation({
            "event_id": "close_point",
            "device_id": "s1",
            "latitude": center_lat + 0.002,  # ~222 meters
            "longitude": center_lon,
            "noise_level_db": 65.0,
            "sound_class": "drilling",
            "confidence": 0.8,
            "calibrated_db": 65.0,
        })
        # Point ~5 km away
        repo.insert_observation({
            "event_id": "far_point",
            "device_id": "s2",
            "latitude": center_lat + 0.045,  # ~5,000 meters
            "longitude": center_lon,
            "noise_level_db": 70.0,
            "sound_class": "drilling",
            "confidence": 0.8,
            "calibrated_db": 70.0,
        })

        results = repo.get_observations_in_radius(center_lat, center_lon, radius_meters=1000.0)
        assert len(results) == 1
        assert results[0].event_id == "close_point"

    def test_sound_class_filter_and_counts(self, repo):
        for i in range(3):
            repo.insert_observation({
                "event_id": f"siren_{i}",
                "device_id": "s1",
                "latitude": 18.52,
                "longitude": 73.85,
                "noise_level_db": 80.0,
                "sound_class": "siren",
                "confidence": 0.9,
                "calibrated_db": 80.0,
            })
        repo.insert_observation({
            "event_id": "jackhammer_0",
            "device_id": "s1",
            "latitude": 18.52,
            "longitude": 73.85,
            "noise_level_db": 90.0,
            "sound_class": "jackhammer",
            "confidence": 0.9,
            "calibrated_db": 90.0,
        })

        sirens = repo.get_observations_by_class("siren")
        assert len(sirens) == 3

        counts = repo.get_sound_event_counts()
        assert counts["siren"] == 3
        assert counts["jackhammer"] == 1

    def test_area_noise_summary_and_baseline(self, repo):
        for db_val in [60.0, 70.0, 80.0]:
            repo.insert_observation({
                "event_id": f"ev_{db_val}",
                "device_id": "s1",
                "latitude": 18.52,
                "longitude": 73.85,
                "noise_level_db": db_val,
                "sound_class": "ambient",
                "confidence": 0.9,
                "calibrated_db": db_val,
            })

        summary = repo.get_average_noise_by_area(18.50, 73.80, 18.55, 73.90)
        assert summary["average_db"] == 70.0
        assert summary["min_db"] == 60.0
        assert summary["max_db"] == 80.0
        assert summary["count"] == 3

        baseline = repo.get_historical_baseline(time_window_hours=24)
        assert baseline["mean_db"] == 70.0
        assert baseline["std_db"] == 10.0
        assert baseline["observation_count"] == 3
