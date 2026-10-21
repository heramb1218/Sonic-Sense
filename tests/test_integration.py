"""
End-to-End System Integration Tests (Phase 12)

Comprehensive verification of the unified pipeline across all 9 subsystems:
1. Audio Preprocessing & Spectrogram Extraction
2. PyTorch CNN Deep Learning Inference
3. Sensor Mobility, Calibration & Event Packaging
4. Message Broker (Kafka/LocalQueue) Streaming Publication
5. Consumer Ingestion & FIFO Processing
6. Multi-Factor Acoustic Anomaly Detection
7. Spatial-Temporal Database Persistence (PostGIS/SQLite)
8. Uber H3 Hexagonal Spatial Aggregation & GeoJSON Generation
9. REST API & Real-Time WebSocket Map Broadcast
"""

import uuid
from datetime import datetime, timezone
from pathlib import Path
import pytest
import numpy as np
from fastapi.testclient import TestClient

from src.config.settings import settings
from src.audio.processor import load_audio, preprocess_audio
from src.features.extractor import extract_log_mel_spectrogram
from src.inference.predict import SoundClassifier
from src.simulation.sensors import SimulatedSensor
from src.simulation.zones import ZoneRegistry
from src.streaming.broker import get_broker
from src.streaming.producer import NoiseEventProducer
from src.streaming.consumer import NoiseEventConsumer
from src.anomaly.detector import AcousticAnomalyDetector
from src.database.connection import get_db_manager
from src.database.repository import ObservationRepository
from src.spatial.h3_grid import H3GridManager
from backend.main import app


@pytest.fixture(scope="module")
def client():
    """FastAPI TestClient fixture."""
    return TestClient(app)


@pytest.fixture(scope="module")
def db_repo():
    """Database repository fixture initialized for integration tests."""
    db = get_db_manager()
    db.init_db()
    return ObservationRepository(db_manager=db)


@pytest.fixture(scope="module")
def sample_audio_path():
    """Fixture ensuring a valid sample audio file is available."""
    sample_path = settings.DATA_DIR / "sample" / "sample_siren_urban.wav"
    if not sample_path.exists():
        import soundfile as sf
        from scripts.generate_sample_audio import generate_synthetic_siren
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        syn_audio = generate_synthetic_siren(duration=settings.AUDIO_DURATION, sr=settings.SAMPLE_RATE)
        sf.write(str(sample_path), syn_audio, settings.SAMPLE_RATE)
    return sample_path


class TestEndToEndSystemIntegration:
    """End-to-End System Integration Test Suite covering complete acoustic pipeline."""

    def test_audio_to_cnn_inference(self, sample_audio_path):
        """Verify audio loading, preprocessing, spectrogram, and CNN prediction."""
        audio, sr = load_audio(sample_audio_path)
        assert len(audio) > 0
        assert sr == settings.SAMPLE_RATE

        proc_audio = preprocess_audio(audio, sr=sr)
        log_mel = extract_log_mel_spectrogram(proc_audio, sr=sr)

        assert log_mel.ndim == 2
        assert log_mel.shape[0] == 128
        assert log_mel.shape[1] > 100

        classifier = SoundClassifier()
        pred = classifier.predict_audio(proc_audio, sr=sr)

        assert "predicted_class" in pred
        assert "confidence" in pred
        assert 0.0 <= pred["confidence"] <= 1.0
        assert pred["predicted_class"] in classifier.classes.values()

    def test_sensor_event_generation_and_mobility(self, sample_audio_path):
        """Verify simulated sensor calibration, movement, and observation creation."""
        registry = ZoneRegistry()
        zone = registry.get_all_zones()[0]
        sensor = SimulatedSensor(
            device_id="INTEG-SENSOR-001",
            device_model="Samsung Galaxy S23",
            calibration_offset=1.2,
            initial_lat=zone.center_lat,
            initial_lon=zone.center_lon,
            current_zone=zone,
        )

        initial_lat, initial_lon = sensor.lat, sensor.lon
        sensor.step(dt_seconds=5.0)

        # Coordinate should have updated with mobility
        assert (sensor.lat != initial_lat) or (sensor.lon != initial_lon)

        classifier = SoundClassifier()
        event = sensor.capture_observation(
            audio_file_path=sample_audio_path,
            classifier=classifier,
        )

        assert event["device_id"] == "INTEG-SENSOR-001"
        assert event["device_model"] == "Samsung Galaxy S23"
        assert event["calibration_offset"] == 1.2
        assert abs(event["calibrated_db"] - (event["noise_level_db"] + 1.2)) < 1e-4
        assert event["h3_cell"] is not None
        assert len(event["h3_cell"]) >= 15

    def test_streaming_broker_to_consumer_flow(self):
        """Verify topic creation, serialization, message broker publish, and consumer ingestion."""
        broker = get_broker(force_mode="local")
        topic = f"integ-topic-{uuid.uuid4().hex[:6]}"
        producer = NoiseEventProducer(broker=broker, topic=topic)

        test_payload = {
            "event_id": f"EVT-{uuid.uuid4().hex[:8]}",
            "device_id": "TEST-BROKER-01",
            "noise_level_db": 78.5,
            "sound_class": "drilling",
            "confidence": 0.88,
        }

        msg = producer.send_event(test_payload, verbose=False)
        assert msg is not None
        assert msg.topic == topic

        consumer = NoiseEventConsumer(broker=broker, topic=topic)
        consumed = list(consumer.consume_events(max_messages=1, timeout=0.5, verbose=False))
        assert len(consumed) == 1
        assert consumed[0]["event_id"] == test_payload["event_id"]
        assert consumed[0]["sound_class"] == "drilling"

    def test_anomaly_detection_engine(self):
        """Verify acoustic anomaly scoring, severity thresholds, and explanation breakdown."""
        detector = AcousticAnomalyDetector()

        # Severe anomaly test case (siren at high dB in quiet residential area)
        severe_obs = {
            "latitude": 18.5204,
            "longitude": 73.8567,
            "calibrated_db": 95.0,
            "sound_class": "siren",
            "confidence": 0.95,
        }

        eval_res = detector.evaluate_observation(severe_obs)
        assert eval_res["severity"] in ("HIGH", "CRITICAL")
        assert eval_res["anomaly_score"] > 2.0
        assert len(eval_res["reasons"]) >= 1

    def test_database_persistence_and_h3_aggregation(self, db_repo):
        """Verify observation storage in DB and Uber H3 hexagonal aggregation."""
        unique_id = f"INTEG-DB-{uuid.uuid4().hex[:8]}"
        obs_payload = {
            "event_id": unique_id,
            "device_id": "SENSOR-AGG-01",
            "device_model": "Google Pixel 8",
            "timestamp": "2026-10-15T00:00:00Z",
            "latitude": 18.5300,
            "longitude": 73.8500,
            "gps_accuracy": 4.0,
            "noise_level_db": 82.0,
            "sound_class": "jackhammer",
            "confidence": 0.91,
            "calibration_offset": 0.0,
            "calibrated_db": 82.0,
            "h3_cell": "8860885017fffff",
            "anomaly_score": 2.5,
        }

        saved_obs = db_repo.insert_observation(obs_payload)
        assert saved_obs.id is not None
        assert saved_obs.event_id == unique_id

        # Query latest records
        recent_records = db_repo.get_latest_observations(limit=10)
        assert len(recent_records) >= 1

        # Aggregate with H3
        hexagons = H3GridManager.aggregate_acoustic_observations(recent_records, resolution=8)
        assert len(hexagons) >= 1

        # Verify GeoJSON conversion
        geojson = H3GridManager.to_geojson_feature_collection(hexagons)
        assert geojson["type"] == "FeatureCollection"
        assert len(geojson["features"]) == len(hexagons)
        first_feat = geojson["features"][0]
        assert "geometry" in first_feat
        assert first_feat["geometry"]["type"] == "Polygon"
        assert "properties" in first_feat
        assert "average_db" in first_feat["properties"]

    def test_full_pipeline_rest_to_websocket_delivery(self, client):
        """Verify POST /observations triggers real-time WebSocket broadcast and updates endpoints."""
        event_id = f"WS-INTEG-{uuid.uuid4().hex[:8]}"
        payload = {
            "event_id": event_id,
            "device_id": "SENSOR-REST-WS-01",
            "device_model": "OnePlus 11",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "latitude": 18.5250,
            "longitude": 73.8550,
            "gps_accuracy": 3.5,
            "noise_level_db": 74.0,
            "sound_class": "car_horn",
            "confidence": 0.89,
            "calibration_offset": -1.0,
            "calibrated_db": 73.0,
            "h3_cell": "8860885017fffff",
            "anomaly_score": 1.2,
        }

        with client.websocket_connect("/ws/live") as ws:
            post_resp = client.post("/observations", json=payload)
            assert post_resp.status_code == 201

            broadcast_msg = ws.receive_json()
            assert broadcast_msg["event_id"] == event_id
            assert broadcast_msg["sound_class"] == "car_horn"
            assert broadcast_msg["calibrated_db"] == 73.0

        # Query REST API endpoints to confirm state reflects the event
        obs_resp = client.get("/observations/latest?limit=10")
        assert obs_resp.status_code == 200
        found = any(o["event_id"] == event_id for o in obs_resp.json())
        assert found

        stats_resp = client.get("/statistics")
        assert stats_resp.status_code == 200
        assert stats_resp.json()["total_observations"] > 0
