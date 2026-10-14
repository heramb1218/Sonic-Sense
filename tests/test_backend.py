"""
Unit and Integration Tests for FastAPI Backend (Phase 7)

Tests:
- /health endpoint status and database dialect
- /sound-classes metadata taxonomy
- POST /observations ingestion and persistence
- GET /observations (default, filtered by class, filtered by bounding box)
- GET /observations/latest
- GET /heatmap spatial intensity generation
- GET /hexagons cell-level acoustic aggregations
- GET /statistics global analytics
- GET /statistics/area spatial aggregation
- GET /anomalies query
- WebSocket /ws/live connectivity, ping-pong, and real-time event broadcasting
"""

import uuid
from datetime import datetime, timezone
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from src.database.connection import get_db_manager
from src.database.repository import ObservationRepository


@pytest.fixture(scope="module")
def client():
    """Create a FastAPI test client instance."""
    return TestClient(app)


@pytest.fixture(scope="module")
def repo():
    """Provide ObservationRepository instance."""
    db = get_db_manager()
    db.init_db()
    return ObservationRepository(db_manager=db)


@pytest.fixture
def sample_event():
    """Generate a valid test observation payload."""
    return {
        "event_id": str(uuid.uuid4()),
        "device_id": "TEST-SENSOR-001",
        "device_model": "PyTest-Virtual-Acoustic",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "latitude": 40.7128,
        "longitude": -74.0060,
        "gps_accuracy": 3.5,
        "noise_level_db": 82.5,
        "sound_class": "siren",
        "confidence": 0.94,
        "calibration_offset": -2.0,
        "calibrated_db": 80.5,
        "h3_cell": "882a100d2bffff",
        "anomaly_score": 2.8,
    }


def test_health_check(client):
    """Test /health system status endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "database_engine" in data
    assert "timestamp" in data
    assert data["active_connections"] >= 0


def test_sound_classes_endpoint(client):
    """Test /sound-classes taxonomy catalog."""
    response = client.get("/sound-classes")
    assert response.status_code == 200
    data = response.json()
    assert data["total_classes"] == 10
    classes = data["classes"]
    # Keys might be strings or ints in JSON response
    class_names = list(classes.values())
    assert "siren" in class_names
    assert "drilling" in class_names
    assert "street_music" in class_names


def test_post_observation(client, sample_event):
    """Test POST /observations ingest endpoint."""
    response = client.post("/observations", json=sample_event)
    assert response.status_code == 201
    data = response.json()
    assert data["event_id"] == sample_event["event_id"]
    assert data["sound_class"] == "siren"
    assert data["calibrated_db"] == 80.5
    assert data["latitude"] == 40.7128


def test_get_observations(client):
    """Test GET /observations list retrieval."""
    response = client.get("/observations?limit=10")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 1


def test_get_latest_observations(client):
    """Test GET /observations/latest endpoint."""
    response = client.get("/observations/latest?limit=5")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) <= 5


def test_filter_observations_by_class(client, sample_event):
    """Test GET /observations filtered by sound category."""
    # Ensure our sample event is in DB
    client.post("/observations", json=sample_event)

    response = client.get("/observations?sound_class=siren&limit=10")
    assert response.status_code == 200
    records = response.json()
    assert len(records) > 0
    for r in records:
        assert r["sound_class"] == "siren"


def test_filter_observations_by_bbox(client, sample_event):
    """Test GET /observations filtered by spatial bounding box."""
    client.post("/observations", json=sample_event)

    response = client.get(
        "/observations?min_lat=40.0&min_lon=-75.0&max_lat=41.5&max_lon=-73.0&limit=10"
    )
    assert response.status_code == 200
    records = response.json()
    assert len(records) > 0
    for r in records:
        assert 40.0 <= r["latitude"] <= 41.5
        assert -75.0 <= r["longitude"] <= -73.0


def test_heatmap_endpoint(client):
    """Test GET /heatmap points generation."""
    response = client.get("/heatmap?limit=50")
    assert response.status_code == 200
    points = response.json()
    assert isinstance(points, list)
    if points:
        p = points[0]
        assert "latitude" in p
        assert "longitude" in p
        assert "intensity" in p
        assert 0.05 <= p["intensity"] <= 1.0
        assert "calibrated_db" in p


def test_hexagons_endpoint(client):
    """Test GET /hexagons spatial aggregation."""
    response = client.get("/hexagons?resolution=8")
    assert response.status_code == 200
    cells = response.json()
    assert isinstance(cells, list)
    if cells:
        c = cells[0]
        assert "h3_cell" in c
        assert "average_db" in c
        assert "dominant_sound_class" in c
        assert "observation_count" in c


def test_statistics_endpoint(client):
    """Test GET /statistics global metrics."""
    response = client.get("/statistics")
    assert response.status_code == 200
    data = response.json()
    assert data["total_observations"] >= 1
    assert isinstance(data["sound_class_distribution"], dict)
    assert "baseline_mean_db" in data
    assert "baseline_std_db" in data
    assert data["active_devices_count"] >= 1


def test_statistics_area_endpoint(client):
    """Test GET /statistics/area regional aggregation."""
    response = client.get(
        "/statistics/area?min_lat=40.0&min_lon=-75.0&max_lat=41.5&max_lon=-73.0"
    )
    assert response.status_code == 200
    data = response.json()
    assert "average_db" in data
    assert "max_db" in data
    assert "min_db" in data
    assert data["count"] >= 1


def test_anomalies_endpoint(client):
    """Test GET /anomalies retrieval."""
    response = client.get("/anomalies?min_score=1.5&limit=20")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


def test_websocket_ping_pong(client):
    """Test WebSocket connection and ping-pong interaction."""
    with client.websocket_connect("/ws/live") as websocket:
        websocket.send_text("ping")
        data = websocket.receive_text()
        assert data == "pong"


def test_websocket_broadcast_on_post(client, sample_event):
    """Test real-time event broadcasting to connected WebSocket clients when an observation is posted."""
    with client.websocket_connect("/ws/live") as websocket:
        post_response = client.post("/observations", json=sample_event)
        assert post_response.status_code == 201

        # Receive broadcasted event payload
        event_data = websocket.receive_json()
        assert event_data["event_id"] == sample_event["event_id"]
        assert event_data["sound_class"] == sample_event["sound_class"]
        assert event_data["calibrated_db"] == sample_event["calibrated_db"]
