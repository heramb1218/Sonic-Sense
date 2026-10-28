"""
Unit and Integration Tests for FastAPI Backend (Phase 7 & Phase 1 Regression)

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
- Database isolation regression test (zero writes to urban_noise.db)
- Strict request validation for POST /observations (bounds, required fields, HTTP 422)
"""

import uuid
from datetime import datetime, timezone
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.dependencies import get_repository
from src.config.settings import settings
from src.database.connection import DatabaseManager
from src.database.repository import ObservationRepository
from src.database.models import NoiseObservation


@pytest.fixture(scope="module")
def test_db():
    """Create an isolated in-memory DatabaseManager for testing."""
    manager = DatabaseManager(db_url="sqlite:///:memory:")
    yield manager
    manager.engine.dispose()


@pytest.fixture(scope="module")
def repo(test_db):
    """Provide ObservationRepository instance using isolated in-memory DB."""
    return ObservationRepository(db_manager=test_db)


@pytest.fixture(scope="module")
def client(repo):
    """Create a FastAPI test client instance with isolated test DB override."""
    app.dependency_overrides[get_repository] = lambda: repo
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


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
    assert len(points) > 0
    first = points[0]
    assert "latitude" in first
    assert "longitude" in first
    assert "intensity" in first
    assert 0.05 <= first["intensity"] <= 1.0


def test_hexagons_endpoint(client):
    """Test GET /hexagons acoustic spatial aggregation."""
    response = client.get("/hexagons?resolution=8&limit=50&geojson=false")
    assert response.status_code == 200
    cells = response.json()
    assert isinstance(cells, list)
    assert len(cells) > 0
    first = cells[0]
    assert "h3_cell" in first
    assert "average_db" in first
    assert "observation_count" in first


def test_statistics_endpoint(client):
    """Test GET /statistics summary KPIs."""
    response = client.get("/statistics")
    assert response.status_code == 200
    data = response.json()
    assert data["total_observations"] > 0
    assert "sound_class_distribution" in data
    assert isinstance(data["sound_class_distribution"], dict)
    assert "baseline_mean_db" in data
    assert "active_devices_count" in data


def test_statistics_area_endpoint(client):
    """Test GET /statistics/area spatial aggregation query."""
    response = client.get(
        "/statistics/area?min_lat=40.0&min_lon=-75.0&max_lat=41.5&max_lon=-73.0"
    )
    assert response.status_code == 200
    data = response.json()
    assert "average_db" in data
    assert "max_db" in data
    assert "min_db" in data
    assert "count" in data
    assert data["count"] > 0


def test_anomalies_endpoint(client, sample_event):
    """Test GET /anomalies threshold filtering."""
    # Ensure event with anomaly_score 2.8 exists
    client.post("/observations", json=sample_event)

    response = client.get("/anomalies?min_score=2.0&limit=10")
    assert response.status_code == 200
    anomalies = response.json()
    assert len(anomalies) > 0
    for a in anomalies:
        assert a["anomaly_score"] >= 2.0


def test_websocket_ping_pong(client):
    """Test WebSocket heartbeat ping-pong exchange."""
    with client.websocket_connect("/ws/live") as websocket:
        websocket.send_text("ping")
        data = websocket.receive_text()
        assert data == "pong"


def test_websocket_broadcast_on_post(client, sample_event):
    """Test that POST /observations triggers broadcast to connected WebSockets."""
    with client.websocket_connect("/ws/live") as websocket:
        # Create new observation
        response = client.post("/observations", json=sample_event)
        assert response.status_code == 201

        # Receive broadcasted event payload
        event_data = websocket.receive_json()
        assert event_data["event_id"] == sample_event["event_id"]
        assert event_data["sound_class"] == sample_event["sound_class"]
        assert event_data["calibrated_db"] == sample_event["calibrated_db"]


def test_frontend_root_html(client):
    """Test that GET / serves the interactive frontend HTML dashboard."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "Urban Acoustic Intelligence" in response.text
    assert "map-viewport" in response.text
    assert "sound-class-chart" in response.text


def test_frontend_css_static(client):
    """Test serving CSS static stylesheet."""
    response = client.get("/css/style.css")
    assert response.status_code == 200
    assert "text/css" in response.headers.get("content-type", "")
    assert "--primary-cyan" in response.text


def test_frontend_js_static(client):
    """Test serving JavaScript frontend app entrypoint."""
    response = client.get("/js/app.js")
    assert response.status_code == 200
    assert "AcousticDashboardApp" in response.text


# =============================================================================
# Phase 1 Regression & Validation Tests
# =============================================================================

def test_database_isolation_does_not_modify_development_db(client, repo, sample_event):
    """Regression test: verify test requests do not write to the development database."""
    # Check current row count in the real development database
    dev_db = DatabaseManager(db_url=settings.DATABASE_URL)
    with dev_db.session_scope() as dev_session:
        initial_dev_count = dev_session.query(NoiseObservation).count()

    # Ingest a new observation via test client
    isolated_event = dict(sample_event)
    isolated_event["event_id"] = f"ISOLATION-REGRESSION-{uuid.uuid4().hex[:6]}"
    res = client.post("/observations", json=isolated_event)
    assert res.status_code == 201

    # Verify event exists in the isolated test repository
    test_obs = repo.get_latest_observations(limit=10)
    assert any(o.event_id == isolated_event["event_id"] for o in test_obs)

    # Verify row count in the real development database did NOT increase
    with dev_db.session_scope() as dev_session:
        final_dev_count = dev_session.query(NoiseObservation).count()
        assert final_dev_count == initial_dev_count, (
            f"Database isolation violated: dev DB row count increased from {initial_dev_count} to {final_dev_count}"
        )


def test_post_observation_valid_defaults(client):
    """Test POST /observations with minimal required fields (defaults computed)."""
    minimal_payload = {
        "device_id": "MINIMAL-SENSOR-01",
        "latitude": 18.5204,
        "longitude": 73.8567,
        "noise_level_db": 65.0,
        "sound_class": "engine_idling",
        "confidence": 0.88,
    }
    response = client.post("/observations", json=minimal_payload)
    assert response.status_code == 201
    data = response.json()
    assert data["event_id"] is not None
    assert data["calibrated_db"] == 65.0
    assert data["timestamp"] is not None
    assert data["sound_class"] == "engine_idling"


def test_post_observation_rejects_invalid_latitude(client, sample_event):
    """Test POST /observations rejects latitude out of [-90, 90] bounds."""
    bad_payload = dict(sample_event)
    bad_payload["latitude"] = 91.5
    response = client.post("/observations", json=bad_payload)
    assert response.status_code == 422
    assert "latitude" in response.text


def test_post_observation_rejects_invalid_longitude(client, sample_event):
    """Test POST /observations rejects longitude out of [-180, 180] bounds."""
    bad_payload = dict(sample_event)
    bad_payload["longitude"] = -185.0
    response = client.post("/observations", json=bad_payload)
    assert response.status_code == 422
    assert "longitude" in response.text


def test_post_observation_rejects_invalid_decibel_range(client, sample_event):
    """Test POST /observations rejects sound levels outside [0, 150] dB."""
    # Negative decibels
    bad_payload1 = dict(sample_event)
    bad_payload1["noise_level_db"] = -10.0
    res1 = client.post("/observations", json=bad_payload1)
    assert res1.status_code == 422

    # Excessive decibels (>150 dB)
    bad_payload2 = dict(sample_event)
    bad_payload2["noise_level_db"] = 250.0
    res2 = client.post("/observations", json=bad_payload2)
    assert res2.status_code == 422


def test_post_observation_rejects_invalid_confidence(client, sample_event):
    """Test POST /observations rejects confidence outside [0.0, 1.0]."""
    bad_payload = dict(sample_event)
    bad_payload["confidence"] = 1.5
    response = client.post("/observations", json=bad_payload)
    assert response.status_code == 422
    assert "confidence" in response.text


def test_post_observation_rejects_missing_required_fields(client):
    """Test POST /observations rejects empty or missing required fields with HTTP 422."""
    response = client.post("/observations", json={})
    assert response.status_code == 422
