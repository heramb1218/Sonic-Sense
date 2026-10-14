"""
Observation Repository — Spatial-Temporal Database Queries (Section 20)

Implements the complete set of required spatial-temporal queries:
1. Insert single observation & batch insert
2. Get latest observations
3. Get observations in time range
4. Get observations within geographic bounding box
5. Get observations within radial distance (Haversine filtering)
6. Get observations filtered by sound class
7. Get average noise level and summary metrics by spatial area
8. Get sound event counts by category
9. Get historical baseline (mean, standard deviation, count)
10. Get acoustic anomalies above threshold
"""

import math
from datetime import datetime, timezone, timedelta
from typing import Optional, Sequence, Union
from sqlalchemy import func, desc, and_

from src.database.connection import DatabaseManager, get_db_manager
from src.database.models import NoiseObservation
from src.spatial.h3_grid import H3GridManager


class ObservationRepository:
    """Repository handling all spatial-temporal data queries on noise_observations."""

    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self.db = db_manager or get_db_manager()

    def insert_observation(self, event: dict) -> NoiseObservation:
        """Insert a single acoustic event record."""
        ts = event.get("timestamp")
        if isinstance(ts, str):
            # Parse ISO-8601 string to datetime
            ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        elif ts is None:
            ts = datetime.now(timezone.utc)

        lat = float(event["latitude"])
        lon = float(event["longitude"])
        h3_cell = event.get("h3_cell") or H3GridManager.lat_lng_to_cell(lat, lon)

        obs = NoiseObservation(
            event_id=event["event_id"],
            device_id=event["device_id"],
            device_model=event.get("device_model"),
            timestamp=ts,
            latitude=lat,
            longitude=lon,
            gps_accuracy=event.get("gps_accuracy"),
            noise_level_db=float(event["noise_level_db"]),
            sound_class=str(event["sound_class"]),
            confidence=float(event["confidence"]),
            calibration_offset=float(event.get("calibration_offset", 0.0)),
            calibrated_db=float(event["calibrated_db"]),
            h3_cell=h3_cell,
            anomaly_score=event.get("anomaly_score"),
            audio_file=event.get("audio_file"),
        )

        with self.db.session_scope() as session:
            session.add(obs)
            session.flush()
            # Detach/refresh
            session.refresh(obs)
            session.expunge(obs)
            return obs

    def insert_batch(self, events: list[dict]) -> int:
        """Insert multiple observation events in a single transaction."""
        count = 0
        with self.db.session_scope() as session:
            for ev in events:
                ts = ev.get("timestamp")
                if isinstance(ts, str):
                    ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                elif ts is None:
                    ts = datetime.now(timezone.utc)

                lat = float(ev["latitude"])
                lon = float(ev["longitude"])
                h3_cell = ev.get("h3_cell") or H3GridManager.lat_lng_to_cell(lat, lon)

                obs = NoiseObservation(
                    event_id=ev["event_id"],
                    device_id=ev["device_id"],
                    device_model=ev.get("device_model"),
                    timestamp=ts,
                    latitude=lat,
                    longitude=lon,
                    gps_accuracy=ev.get("gps_accuracy"),
                    noise_level_db=float(ev["noise_level_db"]),
                    sound_class=str(ev["sound_class"]),
                    confidence=float(ev["confidence"]),
                    calibration_offset=float(ev.get("calibration_offset", 0.0)),
                    calibrated_db=float(ev["calibrated_db"]),
                    h3_cell=h3_cell,
                    anomaly_score=ev.get("anomaly_score"),
                    audio_file=ev.get("audio_file"),
                )
                session.add(obs)
                count += 1
        return count

    # =========================================================================
    # Query 1: Get latest observations
    # =========================================================================
    def get_latest_observations(self, limit: int = 50) -> list[NoiseObservation]:
        """Fetch the most recent N noise observations sorted by timestamp descending."""
        with self.db.session_scope() as session:
            records = (
                session.query(NoiseObservation)
                .order_by(desc(NoiseObservation.timestamp))
                .limit(limit)
                .all()
            )
            for r in records:
                session.expunge(r)
            return records

    # =========================================================================
    # Query 2: Get observations in a time range
    # =========================================================================
    def get_observations_in_time_range(
        self,
        start_time: datetime,
        end_time: datetime,
        limit: int = 500,
    ) -> list[NoiseObservation]:
        """Fetch observations recorded between start_time and end_time."""
        with self.db.session_scope() as session:
            records = (
                session.query(NoiseObservation)
                .filter(and_(
                    NoiseObservation.timestamp >= start_time,
                    NoiseObservation.timestamp <= end_time,
                ))
                .order_by(NoiseObservation.timestamp.asc())
                .limit(limit)
                .all()
            )
            for r in records:
                session.expunge(r)
            return records

    # =========================================================================
    # Query 3: Get observations within a geographic area (Bounding Box)
    # =========================================================================
    def get_observations_in_bounding_box(
        self,
        min_lat: float,
        min_lon: float,
        max_lat: float,
        max_lon: float,
        limit: int = 500,
    ) -> list[NoiseObservation]:
        """Spatial bounding box query leveraging the compound coordinate index."""
        with self.db.session_scope() as session:
            records = (
                session.query(NoiseObservation)
                .filter(and_(
                    NoiseObservation.latitude >= min_lat,
                    NoiseObservation.latitude <= max_lat,
                    NoiseObservation.longitude >= min_lon,
                    NoiseObservation.longitude <= max_lon,
                ))
                .limit(limit)
                .all()
            )
            for r in records:
                session.expunge(r)
            return records

    # =========================================================================
    # Query 4: Get observations within a radial distance (Radius query)
    # =========================================================================
    def get_observations_in_radius(
        self,
        center_lat: float,
        center_lon: float,
        radius_meters: float,
        limit: int = 500,
    ) -> list[NoiseObservation]:
        """Radial search centered around a point."""
        # 1 deg latitude ≈ 111,111 meters
        delta_lat = radius_meters / 111111.0
        delta_lon = radius_meters / (111111.0 * max(0.1, math.cos(math.radians(center_lat))))

        # Fast pre-filtering using indexed bounding box
        candidates = self.get_observations_in_bounding_box(
            min_lat=center_lat - delta_lat,
            min_lon=center_lon - delta_lon,
            max_lat=center_lat + delta_lat,
            max_lon=center_lon + delta_lon,
            limit=limit * 2,
        )

        # Precise Haversine distance post-filtering
        results = []
        for obs in candidates:
            dist = self._haversine(center_lat, center_lon, obs.latitude, obs.longitude)
            if dist <= radius_meters:
                results.append(obs)
                if len(results) >= limit:
                    break
        return results

    # =========================================================================
    # Query 5: Get observations for a particular sound class
    # =========================================================================
    def get_observations_by_class(
        self,
        sound_class: str,
        limit: int = 100,
    ) -> list[NoiseObservation]:
        """Filter observations by sound class category."""
        with self.db.session_scope() as session:
            records = (
                session.query(NoiseObservation)
                .filter(NoiseObservation.sound_class == sound_class)
                .order_by(desc(NoiseObservation.timestamp))
                .limit(limit)
                .all()
            )
            for r in records:
                session.expunge(r)
            return records

    # =========================================================================
    # Query 6: Get average noise level by spatial area
    # =========================================================================
    def get_average_noise_by_area(
        self,
        min_lat: float,
        min_lon: float,
        max_lat: float,
        max_lon: float,
    ) -> dict:
        """Calculate summary statistics (avg, max, min, count) for an area."""
        with self.db.session_scope() as session:
            res = (
                session.query(
                    func.avg(NoiseObservation.calibrated_db).label("avg_db"),
                    func.max(NoiseObservation.calibrated_db).label("max_db"),
                    func.min(NoiseObservation.calibrated_db).label("min_db"),
                    func.count(NoiseObservation.id).label("count"),
                )
                .filter(and_(
                    NoiseObservation.latitude >= min_lat,
                    NoiseObservation.latitude <= max_lat,
                    NoiseObservation.longitude >= min_lon,
                    NoiseObservation.longitude <= max_lon,
                ))
                .first()
            )

            return {
                "average_db": round(float(res.avg_db), 1) if res.avg_db is not None else 0.0,
                "max_db": round(float(res.max_db), 1) if res.max_db is not None else 0.0,
                "min_db": round(float(res.min_db), 1) if res.min_db is not None else 0.0,
                "count": int(res.count or 0),
            }

    # =========================================================================
    # Query 7: Get sound event counts
    # =========================================================================
    def get_sound_event_counts(self) -> dict[str, int]:
        """Aggregate total observations per sound class."""
        with self.db.session_scope() as session:
            rows = (
                session.query(
                    NoiseObservation.sound_class,
                    func.count(NoiseObservation.id),
                )
                .group_by(NoiseObservation.sound_class)
                .all()
            )
            return {cls_name: int(count) for cls_name, count in rows}

    # =========================================================================
    # Query 8: Get historical baseline (for Anomaly Detection in Phase 9)
    # =========================================================================
    def get_historical_baseline(self, time_window_hours: int = 24) -> dict:
        """Calculate historical baseline mean and standard deviation."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=time_window_hours)
        with self.db.session_scope() as session:
            res = (
                session.query(
                    func.avg(NoiseObservation.calibrated_db).label("mean_db"),
                    func.count(NoiseObservation.id).label("total_count"),
                )
                .filter(NoiseObservation.timestamp >= cutoff)
                .first()
            )

            # Standard deviation calculation
            mean_val = float(res.mean_db) if res.mean_db is not None else 0.0
            count = int(res.total_count or 0)

            if count > 1:
                # Fetch values to compute variance (portable across SQLite and Postgres)
                values = [
                    v[0] for v in
                    session.query(NoiseObservation.calibrated_db)
                    .filter(NoiseObservation.timestamp >= cutoff)
                    .all()
                ]
                variance = sum((x - mean_val) ** 2 for x in values) / (count - 1)
                std_val = math.sqrt(variance)
            else:
                std_val = 0.0

            return {
                "mean_db": round(mean_val, 2),
                "std_db": round(std_val, 2),
                "observation_count": count,
                "window_hours": time_window_hours,
            }

    # =========================================================================
    # Query 9: Get anomalies
    # =========================================================================
    def get_anomalies(self, min_score: float = 2.0, limit: int = 100) -> list[NoiseObservation]:
        """Fetch observations flagged with anomaly score exceeding threshold."""
        with self.db.session_scope() as session:
            records = (
                session.query(NoiseObservation)
                .filter(NoiseObservation.anomaly_score >= min_score)
                .order_by(desc(NoiseObservation.timestamp))
                .limit(limit)
                .all()
            )
            for r in records:
                session.expunge(r)
            return records

    # =========================================================================
    # Query 10: Uber H3 Hexagonal Aggregations (Phase 8)
    # =========================================================================
    def get_h3_aggregations(
        self,
        resolution: Optional[int] = None,
        limit: int = 1000,
        anomaly_threshold: float = 2.0,
    ) -> list[dict]:
        """Aggregate recent observations into H3 hexagonal cells with acoustic metrics.

        Returns aggregated statistics including average dB, peak dB, dominant sound class,
        and full GeoJSON Polygon boundaries.
        """
        records = self.get_latest_observations(limit=limit)
        return H3GridManager.aggregate_acoustic_observations(
            records, resolution=resolution, anomaly_threshold=anomaly_threshold
        )

    @staticmethod
    def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate great circle distance in meters."""
        R = 6371000.0
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (
            math.sin(dlat / 2.0) ** 2 +
            math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2.0) ** 2
        )
        return R * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
