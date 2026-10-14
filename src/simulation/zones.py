"""
Urban Geographic Zones — Realistic Spatial Clustering for Acoustic Sensors

Defines distinct functional urban acoustic zones:
1. Residential Area (moderate baseline, domestic sounds)
2. Highway / Traffic Arterial (high baseline, continuous vehicular noise)
3. Commercial / Market Center (moderate-high baseline, varied human sounds)
4. Construction Zone (intermittent loud impulsive & mechanical noise)
5. Quiet Green Zone / Park (low ambient baseline, high acoustic comfort)

All zones are centered around configurable geographical anchors.
"""

from dataclasses import dataclass, field
import math
from typing import Optional
import numpy as np

from src.config.settings import settings


@dataclass
class UrbanZone:
    """Represents an urban spatial zone with distinct acoustic profiles."""
    zone_id: str
    name: str
    zone_type: str  # residential, traffic, commercial, construction, quiet
    center_lat: float
    center_lon: float
    radius_meters: float
    baseline_db_mean: float
    baseline_db_std: float
    typical_classes: list[str] = field(default_factory=list)

    def contains(self, lat: float, lon: float) -> bool:
        """Check if a coordinate falls within this zone's radius."""
        return haversine_distance(self.center_lat, self.center_lon, lat, lon) <= self.radius_meters


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great circle distance between two points in meters."""
    R = 6371000.0  # Earth radius in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2.0) ** 2 +
        math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c


class ZoneRegistry:
    """Registry maintaining urban zones anchored to configured city center."""

    def __init__(self, city_lat: Optional[float] = None, city_lon: Optional[float] = None):
        self.city_lat = city_lat if city_lat is not None else settings.CITY_LAT
        self.city_lon = city_lon if city_lon is not None else settings.CITY_LON
        self.zones: list[UrbanZone] = self._init_default_zones()

    def _init_default_zones(self) -> list[UrbanZone]:
        # 1 deg latitude ≈ 111,000 meters; 0.01 deg ≈ 1,110 meters
        return [
            UrbanZone(
                zone_id="zone_residential_north",
                name="North Residential Colony",
                zone_type="residential",
                center_lat=self.city_lat + 0.012,
                center_lon=self.city_lon - 0.008,
                radius_meters=650.0,
                baseline_db_mean=48.5,
                baseline_db_std=4.0,
                typical_classes=["children_playing", "dog_bark", "air_conditioner"],
            ),
            UrbanZone(
                zone_id="zone_highway_arterial",
                name="Central Highway Express Arterial",
                zone_type="traffic",
                center_lat=self.city_lat - 0.005,
                center_lon=self.city_lon + 0.015,
                radius_meters=850.0,
                baseline_db_mean=76.0,
                baseline_db_std=5.5,
                typical_classes=["engine_idling", "car_horn", "siren"],
            ),
            UrbanZone(
                zone_id="zone_commercial_market",
                name="Downtown Commercial Plaza",
                zone_type="commercial",
                center_lat=self.city_lat,
                center_lon=self.city_lon,
                radius_meters=700.0,
                baseline_db_mean=67.5,
                baseline_db_std=6.0,
                typical_classes=["street_music", "car_horn", "dog_bark"],
            ),
            UrbanZone(
                zone_id="zone_metro_construction",
                name="Metro Transit Construction Site",
                zone_type="construction",
                center_lat=self.city_lat + 0.018,
                center_lon=self.city_lon + 0.010,
                radius_meters=450.0,
                baseline_db_mean=84.5,
                baseline_db_std=7.5,
                typical_classes=["drilling", "jackhammer", "engine_idling"],
            ),
            UrbanZone(
                zone_id="zone_botanical_park",
                name="Urban Botanical Park",
                zone_type="quiet",
                center_lat=self.city_lat - 0.015,
                center_lon=self.city_lon - 0.012,
                radius_meters=550.0,
                baseline_db_mean=40.0,
                baseline_db_std=3.5,
                typical_classes=["ambient", "children_playing", "street_music"],
            ),
        ]

    def get_all_zones(self) -> list[UrbanZone]:
        return self.zones

    def find_zone_for_point(self, lat: float, lon: float) -> Optional[UrbanZone]:
        """Find the zone containing the coordinate, or closest zone."""
        for zone in self.zones:
            if zone.contains(lat, lon):
                return zone
        # Return closest zone if just outside boundaries
        return min(self.zones, key=lambda z: haversine_distance(z.center_lat, z.center_lon, lat, lon))

    def sample_coordinate_in_zone(self, zone: UrbanZone) -> tuple[float, float]:
        """Generate a realistic random coordinate within the given zone."""
        # Random angle and radial distance
        theta = np.random.uniform(0, 2 * math.pi)
        r = zone.radius_meters * math.sqrt(np.random.uniform(0, 1))

        # Convert meters offset to lat/lon degrees
        delta_lat = (r * math.cos(theta)) / 111111.0
        delta_lon = (r * math.sin(theta)) / (111111.0 * math.cos(math.radians(zone.center_lat)))

        return round(zone.center_lat + delta_lat, 6), round(zone.center_lon + delta_lon, 6)
