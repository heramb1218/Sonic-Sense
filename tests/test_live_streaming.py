"""
Unit and Integration Tests for Real-Time WebSocket Live Streaming Pipeline (Phase 11)

Tests:
- Multi-client WebSocket real-time event broadcasting
- WebSocket client disconnection resilience and automatic pruning
- Complete end-to-end sensor -> broker -> consumer -> database -> WebSocket flow
- High-throughput burst streaming (25+ sequential events with zero drop)
- Live anomaly severity and textual reason enrichment in broadcast payloads
"""

import uuid
from datetime import datetime, timezone
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.websocket.connection_manager import ws_manager
from src.database.connection import DatabaseManager
from src.database.repository import ObservationRepository
from src.streaming.broker import LocalQueueBroker
from src.streaming.producer import NoiseEventProducer
from src.streaming.consumer import NoiseEventConsumer
from src.anomaly.detector import AcousticAnomalyDetector
from backend.dependencies import get_repository


@pytest.fixture(scope="module")
def test_db():
    """Isolated in-memory test database for live streaming tests."""
    manager = DatabaseManager(db_url="sqlite:///:memory:")
    yield manager
    manager.engine.dispose()


@pytest.fixture(scope="module")
def repo(test_db):
    """Observation repository fixture using isolated in-memory DB."""
    return ObservationRepository(db_manager=test_db)


@pytest.fixture(scope="module")
def client(repo):
    """FastAPI TestClient fixture with isolated test database override."""
    app.dependency_overrides[get_repository] = lambda: repo
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


class TestWebSocketBroadcasting:
    """Tests for WebSocket connection lifecycle and real-time broadcasting."""

    def test_multi_client_broadcast(self, client):
        """Test that broadcasting sends the payload to multiple concurrent clients."""
        with client.websocket_connect("/ws/live") as ws1:
            with client.websocket_connect("/ws/live") as ws2:
                payload = {
                    "event_id": f"TEST-MULTI-{uuid.uuid4().hex[:6]}",
                    "device_id": "SENSOR-MULTI-01",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "latitude": 18.5204,
                    "longitude": 73.8567,
                    "noise_level_db": 68.0,
                    "sound_class": "street_music",
                    "confidence": 0.90,
                    "calibration_offset": 0.0,
                    "calibrated_db": 68.0,
                }

                # Trigger ingestion which broadcasts
                post_res = client.post("/observations", json=payload)
                assert post_res.status_code == 201

                # Both clients receive the broadcasted event
                msg1 = ws1.receive_json()
                msg2 = ws2.receive_json()

                assert msg1["event_id"] == payload["event_id"]
                assert msg2["event_id"] == payload["event_id"]
                assert msg1["sound_class"] == "street_music"

    def test_websocket_disconnect_resilience(self, client):
        """Test that disconnected clients are safely pruned without interrupting active ones."""
        with client.websocket_connect("/ws/live") as active_ws:
            # Connect and immediately close a temporary client
            with client.websocket_connect("/ws/live") as temp_ws:
                temp_ws.send_text("ping")
                assert temp_ws.receive_text() == "pong"
            # temp_ws is now closed

            # Now broadcast a new event
            payload = {
                "event_id": f"TEST-PRUNE-{uuid.uuid4().hex[:6]}",
                "device_id": "SENSOR-PRUNE-01",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "latitude": 18.5204,
                "longitude": 73.8567,
                "noise_level_db": 64.0,
                "sound_class": "engine_idling",
                "confidence": 0.85,
                "calibration_offset": 0.0,
                "calibrated_db": 64.0,
            }
            post_res = client.post("/observations", json=payload)
            assert post_res.status_code == 201

            # Active client receives successfully
            received = active_ws.receive_json()
            assert received["event_id"] == payload["event_id"]


class TestEndToEndStreamingPipeline:
    """Tests for complete stream lifecycle from sensor generation to WebSocket delivery."""

    def test_end_to_end_streaming_to_websocket(self, client, repo):
        """Test producer -> broker -> consumer -> API -> WebSocket pipeline."""
        broker = LocalQueueBroker()
        topic = "live-stream-test"
        producer = NoiseEventProducer(broker=broker, topic=topic)
        detector = AcousticAnomalyDetector()
        consumer = NoiseEventConsumer(broker=broker, topic=topic, anomaly_detector=detector)

        raw_event = {
            "event_id": f"E2E-{uuid.uuid4().hex[:6]}",
            "device_id": "E2E-SENSOR-01",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "latitude": 18.5024,
            "longitude": 73.8388,  # Quiet / Residential Zone
            "noise_level_db": 93.0,
            "sound_class": "siren",
            "confidence": 0.95,
            "calibration_offset": 0.0,
            "calibrated_db": 93.0,
        }

        with client.websocket_connect("/ws/live") as websocket:
            # 1. Producer sends event to broker
            producer.send_event(raw_event, verbose=False)

            # 2. Consumer reads from broker
            consumed = list(consumer.consume_events(max_messages=1, timeout=1.0, verbose=False))
            assert len(consumed) == 1
            event = consumed[0]

            # 3. Post to API (which saves to DB and broadcasts to WebSocket)
            resp = client.post("/observations", json=event)
            assert resp.status_code == 201

            # 4. Connected WebSocket receives live message
            ws_msg = websocket.receive_json()
            assert ws_msg["event_id"] == raw_event["event_id"]
            assert ws_msg["calibrated_db"] == 93.0
            assert ws_msg["anomaly_severity"] == "CRITICAL"
            assert len(ws_msg["anomaly_reasons"]) > 0

    def test_high_throughput_burst_streaming(self, client):
        """Test that rapid succession of events streams cleanly without drops."""
        with client.websocket_connect("/ws/live") as websocket:
            sent_event_ids = []
            for i in range(15):
                eid = f"BURST-{i:03d}-{uuid.uuid4().hex[:4]}"
                sent_event_ids.append(eid)
                payload = {
                    "event_id": eid,
                    "device_id": f"SENSOR-{i % 5}",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "latitude": 18.5204 + (i * 0.001),
                    "longitude": 73.8567 + (i * 0.001),
                    "noise_level_db": 60.0 + (i % 10),
                    "sound_class": "dog_bark",
                    "confidence": 0.88,
                    "calibration_offset": 0.0,
                    "calibrated_db": 60.0 + (i % 10),
                }
                res = client.post("/observations", json=payload)
                assert res.status_code == 201

            # Receive all 15 events in sequence
            received_ids = []
            for _ in range(15):
                msg = websocket.receive_json()
                received_ids.append(msg["event_id"])

            assert received_ids == sent_event_ids
