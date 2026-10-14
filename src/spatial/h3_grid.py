"""
Uber H3 Spatial Hexagonal Aggregation Engine

Provides hierarchical spatial indexing, boundary GeoJSON generation,
and spatial-temporal acoustic metric aggregation across urban sensor networks.

Key Capabilities:
- Convert (latitude, longitude) coordinates to H3 hexagonal cell indices
- Generate GeoJSON Polygon geometries and FeatureCollections for map rendering
- Compute multi-resolution cell neighbors (k-rings)
- Aggregate acoustic observations by hexagon:
    - Average decibel level (arithmetic and energetic Leq)
    - Peak and minimum decibel levels
    - Dominant sound category and class distribution
    - Anomaly counts based on z-score thresholds
"""

import math
from typing import Sequence, Union, Any, Optional
import h3

from src.config.settings import settings


class H3GridManager:
    """Manages Uber H3 spatial indexing and hexagonal acoustic aggregations."""

    def __init__(self, default_resolution: Optional[int] = None):
        self.default_resolution = (
            default_resolution if default_resolution is not None else settings.H3_DEFAULT_RESOLUTION
        )

    @staticmethod
    def lat_lng_to_cell(lat: float, lon: float, resolution: Optional[int] = None) -> str:
        """Convert GPS coordinates to an H3 hexagonal cell index.

        Args:
            lat: Latitude (-90.0 to 90.0)
            lon: Longitude (-180.0 to 180.0)
            resolution: H3 resolution level (0 to 15, default from settings)

        Returns:
            H3 hexagonal index string (e.g. '882a107289fffff')
        """
        if not (-90.0 <= lat <= 90.0):
            raise ValueError(f"Latitude must be between -90 and 90. Got: {lat}")
        if not (-180.0 <= lon <= 180.0):
            raise ValueError(f"Longitude must be between -180 and 180. Got: {lon}")

        res = resolution if resolution is not None else settings.H3_DEFAULT_RESOLUTION
        if not (0 <= res <= 15):
            raise ValueError(f"H3 resolution must be between 0 and 15. Got: {res}")

        return h3.latlng_to_cell(lat, lon, res)

    @staticmethod
    def cell_to_lat_lng(h3_cell: str) -> tuple[float, float]:
        """Compute the centroid (latitude, longitude) of an H3 cell.

        Args:
            h3_cell: Hexagonal index string

        Returns:
            Tuple of (latitude, longitude)
        """
        lat, lon = h3.cell_to_latlng(h3_cell)
        return round(lat, 6), round(lon, 6)

    @staticmethod
    def cell_to_boundary(h3_cell: str) -> list[tuple[float, float]]:
        """Return the vertex coordinates (lat, lon) defining the hexagonal boundary."""
        boundary = h3.cell_to_boundary(h3_cell)
        return [(round(lat, 6), round(lon, 6)) for lat, lon in boundary]

    @staticmethod
    def cell_to_geojson_geometry(h3_cell: str) -> dict:
        """Generate GeoJSON Polygon geometry for an H3 cell.

        Coordinates follow standard GeoJSON [longitude, latitude] ordering,
        and the ring is explicitly closed by repeating the first vertex.
        """
        boundary = h3.cell_to_boundary(h3_cell)
        # GeoJSON requires [lon, lat] coordinate pairs
        coords = [[round(lon, 6), round(lat, 6)] for lat, lon in boundary]
        # Close the polygon ring
        if coords:
            coords.append(coords[0])
        return {
            "type": "Polygon",
            "coordinates": [coords],
        }

    @classmethod
    def cell_to_geojson_feature(cls, h3_cell: str, properties: Optional[dict] = None) -> dict:
        """Generate a complete GeoJSON Feature for an H3 cell."""
        props = properties.copy() if properties else {}
        props["h3_cell"] = h3_cell
        return {
            "type": "Feature",
            "id": h3_cell,
            "geometry": cls.cell_to_geojson_geometry(h3_cell),
            "properties": props,
        }

    @staticmethod
    def get_cell_neighbors(h3_cell: str, k: int = 1) -> list[str]:
        """Return neighboring cells within k hexagonal steps (k-ring disk)."""
        return list(h3.grid_disk(h3_cell, k))

    @staticmethod
    def is_valid_cell(h3_cell: str) -> bool:
        """Validate whether a string is a legitimate H3 index."""
        try:
            return h3.is_valid_cell(h3_cell)
        except Exception:
            return False

    @staticmethod
    def get_cell_resolution(h3_cell: str) -> int:
        """Return the resolution level of an H3 cell."""
        return h3.get_resolution(h3_cell)

    @classmethod
    def aggregate_acoustic_observations(
        cls,
        observations: Sequence[Union[dict, Any]],
        resolution: Optional[int] = None,
        anomaly_threshold: float = 2.0,
    ) -> list[dict]:
        """Aggregate a collection of acoustic observation records into H3 hexagons.

        Args:
            observations: List of observation dicts or ORM objects with attributes:
                          latitude, longitude, calibrated_db, sound_class, anomaly_score.
            resolution: Target H3 resolution (defaults to settings.H3_DEFAULT_RESOLUTION).
            anomaly_threshold: Minimum z-score to count as an acoustic anomaly.

        Returns:
            List of aggregated cell summary dictionaries.
        """
        res = resolution if resolution is not None else settings.H3_DEFAULT_RESOLUTION
        cells: dict[str, dict] = {}

        for obs in observations:
            # Handle both dictionary payloads and ORM models
            if isinstance(obs, dict):
                lat = obs.get("latitude")
                lon = obs.get("longitude")
                db_val = obs.get("calibrated_db", obs.get("noise_level_db", 0.0))
                s_class = obs.get("sound_class", "unknown")
                anomaly_score = obs.get("anomaly_score")
                h3_id = obs.get("h3_cell")
            else:
                lat = getattr(obs, "latitude", None)
                lon = getattr(obs, "longitude", None)
                db_val = getattr(obs, "calibrated_db", getattr(obs, "noise_level_db", 0.0))
                s_class = getattr(obs, "sound_class", "unknown")
                anomaly_score = getattr(obs, "anomaly_score", None)
                h3_id = getattr(obs, "h3_cell", None)

            if lat is None or lon is None:
                continue

            # Ensure H3 index is valid and matches desired resolution
            if not h3_id or not cls.is_valid_cell(str(h3_id)) or cls.get_cell_resolution(str(h3_id)) != res:
                h3_id = cls.lat_lng_to_cell(float(lat), float(lon), resolution=res)

            if h3_id not in cells:
                center_lat, center_lon = cls.cell_to_lat_lng(h3_id)
                cells[h3_id] = {
                    "h3_cell": h3_id,
                    "resolution": res,
                    "center_lat": center_lat,
                    "center_lon": center_lon,
                    "observation_count": 0,
                    "noise_levels": [],
                    "sound_classes": {},
                    "anomaly_count": 0,
                }

            c = cells[h3_id]
            c["observation_count"] += 1
            c["noise_levels"].append(float(db_val))
            c["sound_classes"][s_class] = c["sound_classes"].get(s_class, 0) + 1

            if anomaly_score is not None and float(anomaly_score) >= anomaly_threshold:
                c["anomaly_count"] += 1

        # Synthesize aggregations
        aggregated_results = []
        for h3_id, data in cells.items():
            levels = data["noise_levels"]
            avg_db = round(sum(levels) / len(levels), 1)

            # Energy-equivalent sound pressure level (Leq):
            # Leq = 10 * log10( (1/N) * sum(10^(Li/10)) )
            energy_sum = sum(math.pow(10.0, l / 10.0) for l in levels)
            energy_avg_db = round(10.0 * math.log10(max(1e-9, energy_sum / len(levels))), 1)

            peak_db = round(max(levels), 1)
            min_db = round(min(levels), 1)

            # Dominant sound class (mode)
            dominant_sound = max(data["sound_classes"].items(), key=lambda x: x[1])[0]

            properties = {
                "h3_cell": h3_id,
                "resolution": data["resolution"],
                "center_lat": data["center_lat"],
                "center_lon": data["center_lon"],
                "observation_count": data["observation_count"],
                "average_db": avg_db,
                "energy_average_db": energy_avg_db,
                "peak_db": peak_db,
                "min_db": min_db,
                "dominant_sound_class": dominant_sound,
                "sound_class_distribution": data["sound_classes"],
                "anomaly_count": data["anomaly_count"],
            }

            geojson_feature = cls.cell_to_geojson_feature(h3_id, properties=properties)

            record = {
                **properties,
                "geojson_feature": geojson_feature,
            }
            aggregated_results.append(record)

        return aggregated_results

    @classmethod
    def to_geojson_feature_collection(cls, aggregated_cells: list[dict]) -> dict:
        """Convert a list of aggregated cell records into a valid GeoJSON FeatureCollection."""
        features = []
        for cell in aggregated_cells:
            if "geojson_feature" in cell:
                features.append(cell["geojson_feature"])
            else:
                features.append(cls.cell_to_geojson_feature(cell["h3_cell"], properties=cell))

        return {
            "type": "FeatureCollection",
            "features": features,
        }
