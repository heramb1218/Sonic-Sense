"""
Unit and Integration Tests for Uber H3 Spatial Hexagonal Aggregation (Phase 8)

Tests:
- GPS coordinate to H3 cell resolution indexing
- Multi-resolution hierarchical indexing (resolutions 7, 8, 9, 10)
- Hexagon boundary polygon and GeoJSON geometry construction
- Coordinate validation and boundary error conditions
- k-ring neighbor search (disk radius)
- Multi-observation acoustic aggregation (mean dB, Leq energy, peak, dominant class, anomalies)
- GeoJSON FeatureCollection generation
- Backend REST API /hexagons endpoint (array and GeoJSON format)
- Automatic H3 cell computation in ObservationRepository
"""

import math
import uuid
from datetime import datetime, timezone
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from src.spatial.h3_grid import H3GridManager
from src.database.connection import get_db_manager
from src.database.repository import ObservationRepository


@pytest.fixture(scope="module")
def client():
    """FastAPI TestClient fixture."""
    return TestClient(app)


@pytest.fixture(scope="module")
def repo():
    """Observation repository fixture."""
    db = get_db_manager()
    db.init_db()
    return ObservationRepository(db_manager=db)


class TestH3GridManager:
    """Unit tests for core H3 spatial functions."""

    def test_lat_lng_to_cell_valid(self):
        """Test valid coordinate conversion to H3 cell index."""
        lat, lon = 40.7128, -74.0060
        cell = H3GridManager.lat_lng_to_cell(lat, lon, resolution=8)

        assert isinstance(cell, str)
        assert len(cell) == 15
        assert H3GridManager.is_valid_cell(cell)
        assert H3GridManager.get_cell_resolution(cell) == 8

    def test_lat_lng_to_cell_invalid_coords(self):
        """Test error handling for out-of-bounds coordinates and resolutions."""
        with pytest.raises(ValueError, match="Latitude"):
            H3GridManager.lat_lng_to_cell(95.0, 0.0)

        with pytest.raises(ValueError, match="Longitude"):
            H3GridManager.lat_lng_to_cell(0.0, 190.0)

        with pytest.raises(ValueError, match="resolution"):
            H3GridManager.lat_lng_to_cell(0.0, 0.0, resolution=16)

    def test_multi_resolution_indexing(self):
        """Test that different resolutions generate distinct hierarchical cell indices."""
        lat, lon = 18.5204, 73.8567
        cell_res7 = H3GridManager.lat_lng_to_cell(lat, lon, resolution=7)
        cell_res8 = H3GridManager.lat_lng_to_cell(lat, lon, resolution=8)
        cell_res9 = H3GridManager.lat_lng_to_cell(lat, lon, resolution=9)

        assert H3GridManager.get_cell_resolution(cell_res7) == 7
        assert H3GridManager.get_cell_resolution(cell_res8) == 8
        assert H3GridManager.get_cell_resolution(cell_res9) == 9
        assert cell_res7 != cell_res8
        assert cell_res8 != cell_res9

    def test_cell_to_lat_lng_centroid(self):
        """Test centroid reconstruction from H3 index."""
        lat, lon = 40.7580, -73.9855
        cell = H3GridManager.lat_lng_to_cell(lat, lon, resolution=8)
        c_lat, c_lon = H3GridManager.cell_to_lat_lng(cell)

        # Centroid should be within ~500m (< 0.01 degrees)
        assert abs(c_lat - lat) < 0.01
        assert abs(c_lon - lon) < 0.01

    def test_cell_to_boundary_and_geometry(self):
        """Test hexagonal boundary vertices and GeoJSON polygon geometry."""
        cell = H3GridManager.lat_lng_to_cell(40.7128, -74.0060, resolution=8)
        boundary = H3GridManager.cell_to_boundary(cell)

        # Hexagon has 6 vertices
        assert len(boundary) == 6

        geom = H3GridManager.cell_to_geojson_geometry(cell)
        assert geom["type"] == "Polygon"
        ring = geom["coordinates"][0]

        # Closed ring: 6 vertices + 1 repeated closing vertex = 7 points
        assert len(ring) == 7
        assert ring[0] == ring[-1]

        # Verify GeoJSON [lon, lat] ordering
        for pt in ring:
            assert len(pt) == 2
            lon, lat = pt[0], pt[1]
            assert -180.0 <= lon <= 180.0
            assert -90.0 <= lat <= 90.0

    def test_cell_neighbors_k_ring(self):
        """Test k-ring neighborhood expansion."""
        cell = H3GridManager.lat_lng_to_cell(40.7128, -74.0060, resolution=8)

        # k=1 ring consists of center cell + 6 immediate neighbors = 7 cells
        neighbors_k1 = H3GridManager.get_cell_neighbors(cell, k=1)
        assert len(neighbors_k1) == 7
        assert cell in neighbors_k1

        # k=2 ring consists of 1 + 6 + 12 = 19 cells
        neighbors_k2 = H3GridManager.get_cell_neighbors(cell, k=2)
        assert len(neighbors_k2) == 19
        assert set(neighbors_k1).issubset(set(neighbors_k2))


class TestAcousticAggregation:
    """Tests for spatial-temporal acoustic aggregation into H3 cells."""

    def test_aggregate_acoustic_observations(self):
        """Test calculating mean dB, peak dB, dominant sound, and anomalies."""
        lat, lon = 40.7128, -74.0060
        observations = [
            {
                "latitude": lat,
                "longitude": lon,
                "calibrated_db": 70.0,
                "sound_class": "engine_idling",
                "anomaly_score": 0.5,
            },
            {
                "latitude": lat + 0.0001,
                "longitude": lon + 0.0001,
                "calibrated_db": 80.0,
                "sound_class": "siren",
                "anomaly_score": 2.5,
            },
            {
                "latitude": lat - 0.0001,
                "longitude": lon - 0.0001,
                "calibrated_db": 90.0,
                "sound_class": "siren",
                "anomaly_score": 3.1,
            },
        ]

        aggregated = H3GridManager.aggregate_acoustic_observations(
            observations, resolution=8, anomaly_threshold=2.0
        )

        assert len(aggregated) == 1
        cell_data = aggregated[0]

        assert cell_data["observation_count"] == 3
        assert cell_data["average_db"] == 80.0  # (70 + 80 + 90) / 3
        assert cell_data["peak_db"] == 90.0
        assert cell_data["min_db"] == 70.0
        assert cell_data["dominant_sound_class"] == "siren"  # 2 sirens vs 1 engine_idling
        assert cell_data["anomaly_count"] == 2  # scores 2.5 and 3.1 >= 2.0
        assert cell_data["energy_average_db"] > 80.0  # Logarithmic Leq is higher than arithmetic mean

    def test_to_geojson_feature_collection(self):
        """Test converting aggregated cells to a GeoJSON FeatureCollection."""
        observations = [
            {"latitude": 40.71, "longitude": -74.00, "calibrated_db": 65.0, "sound_class": "dog_bark"},
            {"latitude": 40.78, "longitude": -73.96, "calibrated_db": 75.0, "sound_class": "traffic"},
        ]
        aggregated = H3GridManager.aggregate_acoustic_observations(observations, resolution=8)
        fc = H3GridManager.to_geojson_feature_collection(aggregated)

        assert fc["type"] == "FeatureCollection"
        assert len(fc["features"]) == len(aggregated)

        feature = fc["features"][0]
        assert feature["type"] == "Feature"
        assert "id" in feature
        assert feature["geometry"]["type"] == "Polygon"
        assert "properties" in feature
        assert "average_db" in feature["properties"]


class TestH3Integration:
    """Integration tests with database repository and FastAPI routes."""

    def test_repository_auto_computes_h3_cell(self, repo):
        """Test that inserting an observation with h3_cell=None automatically populates H3 index."""
        event = {
            "event_id": str(uuid.uuid4()),
            "device_id": "TEST-H3-DEV",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "latitude": 40.730610,
            "longitude": -73.935242,
            "noise_level_db": 72.0,
            "sound_class": "drilling",
            "confidence": 0.88,
            "calibration_offset": 0.0,
            "calibrated_db": 72.0,
            "h3_cell": None,  # Intentionally omitted
        }

        obs = repo.insert_observation(event)
        assert obs.h3_cell is not None
        assert H3GridManager.is_valid_cell(obs.h3_cell)
        assert H3GridManager.get_cell_resolution(obs.h3_cell) == 8

    def test_api_hexagons_endpoint_list(self, client):
        """Test GET /hexagons default response format (list of cells)."""
        response = client.get("/hexagons?resolution=8&limit=100")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        if data:
            c = data[0]
            assert "h3_cell" in c
            assert "average_db" in c
            assert "dominant_sound_class" in c

    def test_api_hexagons_endpoint_geojson(self, client):
        """Test GET /hexagons?geojson=true returns GeoJSON FeatureCollection."""
        response = client.get("/hexagons?resolution=8&limit=100&geojson=true")
        assert response.status_code == 200
        fc = response.json()
        assert fc["type"] == "FeatureCollection"
        assert isinstance(fc["features"], list)
        if fc["features"]:
            f = fc["features"][0]
            assert f["type"] == "Feature"
            assert f["geometry"]["type"] == "Polygon"
            assert "average_db" in f["properties"]
