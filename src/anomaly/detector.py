"""
Acoustic Anomaly Detection Engine (Phase 9)

Implements multi-tiered acoustic anomaly detection:
1. Statistical Z-Score Thresholding against spatial-temporal baseline distributions:
       z = (calibrated_db - baseline_mean) / max(baseline_std, 1.0)
2. Context-Aware Semantic Rules:
       - Municipal nighttime construction violations (22:00 - 06:00)
       - Alarming sound categories (gun_shot, siren, car_horn) in quiet zones
       - Extreme absolute decibel spikes (>= 90 dB SPL)
       - Nighttime residential noise disturbances
3. Unsupervised Machine Learning (scikit-learn Isolation Forest):
       - Multi-feature anomaly scoring on [calibrated_db, duration, confidence, hour_of_day, zone_code]
"""

from datetime import datetime, timezone
from typing import Optional, Sequence, Union, Any
import numpy as np
from sklearn.ensemble import IsolationForest

from src.simulation.zones import ZoneRegistry, UrbanZone


# Default spatial-temporal acoustic baselines: {zone_type: {is_day: (mean_db, std_db)}}
# Day is 07:00 to 22:00; Night is 22:00 to 07:00
DEFAULT_ZONE_BASELINES = {
    "residential": {
        "day": (52.0, 4.5),
        "night": (42.0, 3.5),
    },
    "commercial": {
        "day": (68.0, 5.5),
        "night": (52.0, 4.0),
    },
    "traffic": {
        "day": (76.0, 6.0),
        "night": (62.0, 5.0),
    },
    "construction": {
        "day": (82.0, 7.0),
        "night": (54.0, 4.5),
    },
    "park": {
        "day": (46.0, 3.5),
        "night": (38.0, 2.5),
    },
    "general": {
        "day": (65.0, 6.0),
        "night": (50.0, 5.0),
    },
}

# Zone type integer encoding for ML feature vector
ZONE_TYPE_CODES = {
    "park": 0,
    "residential": 1,
    "commercial": 2,
    "general": 3,
    "traffic": 4,
    "construction": 5,
}


class AcousticAnomalyDetector:
    """Multi-tiered acoustic anomaly detection engine for urban sensor networks."""

    def __init__(
        self,
        z_moderate_threshold: float = 2.0,
        z_critical_threshold: float = 3.0,
        decibel_ceiling: float = 90.0,
        night_start_hour: int = 22,
        night_end_hour: int = 6,
        zone_baselines: Optional[dict] = None,
    ):
        self.z_moderate_threshold = z_moderate_threshold
        self.z_critical_threshold = z_critical_threshold
        self.decibel_ceiling = decibel_ceiling
        self.night_start_hour = night_start_hour
        self.night_end_hour = night_end_hour

        self.baselines = zone_baselines or DEFAULT_ZONE_BASELINES
        self.zone_registry = ZoneRegistry()
        self.ml_model: Optional[IsolationForest] = None
        self._is_ml_fitted: bool = False

    def is_night(self, hour: int) -> bool:
        """Check if hour of day falls within nighttime quiet hours (default: 22:00 - 06:00)."""
        return hour >= self.night_start_hour or hour < self.night_end_hour

    def get_baseline(self, zone_type: str, hour: int) -> tuple[float, float]:
        """Fetch expected mean and standard deviation decibels for zone and hour."""
        normalized_zone = zone_type.lower() if zone_type else "general"
        zone_dict = self.baselines.get(normalized_zone, self.baselines["general"])
        period = "night" if self.is_night(hour) else "day"
        return zone_dict.get(period, (60.0, 5.0))

    def update_baselines_from_data(self, observations: Sequence[Any]) -> dict:
        """Estimate empirical baselines from observed historical records."""
        groups: dict[tuple[str, str], list[float]] = {}

        for obs in observations:
            lat = obs.get("latitude") if isinstance(obs, dict) else getattr(obs, "latitude", None)
            lon = obs.get("longitude") if isinstance(obs, dict) else getattr(obs, "longitude", None)
            db_val = (
                obs.get("calibrated_db", obs.get("noise_level_db"))
                if isinstance(obs, dict)
                else getattr(obs, "calibrated_db", getattr(obs, "noise_level_db", None))
            )
            ts = obs.get("timestamp") if isinstance(obs, dict) else getattr(obs, "timestamp", None)

            if lat is None or lon is None or db_val is None:
                continue

            zone = self.zone_registry.find_zone_for_point(float(lat), float(lon))
            zone_type = zone.zone_type if zone else "general"

            hour = self._extract_hour(ts)
            period = "night" if self.is_night(hour) else "day"

            key = (zone_type, period)
            groups.setdefault(key, []).append(float(db_val))

        # Update baselines with sufficient sample size (N >= 5)
        for (z_type, period), values in groups.items():
            if len(values) >= 5:
                mean_val = float(np.mean(values))
                std_val = float(np.std(values))
                if z_type not in self.baselines:
                    self.baselines[z_type] = {}
                self.baselines[z_type][period] = (round(mean_val, 1), max(1.5, round(std_val, 1)))

        return self.baselines

    def compute_z_score(self, calibrated_db: float, mean_db: float, std_db: float) -> float:
        """Calculate statistical z-score deviation."""
        effective_std = max(float(std_db), 1.0)
        z = (float(calibrated_db) - float(mean_db)) / effective_std
        return round(float(z), 2)

    def check_contextual_rules(
        self,
        calibrated_db: float,
        sound_class: str,
        confidence: float,
        hour: int,
        zone_type: str,
    ) -> list[str]:
        """Evaluate domain-specific contextual rules for acoustic anomalies."""
        reasons = []
        is_nighttime = self.is_night(hour)

        # Rule 1: Extreme decibel ceiling violation
        if calibrated_db >= self.decibel_ceiling:
            reasons.append(f"Extreme acoustic spike ({calibrated_db:.1f} dB >= {self.decibel_ceiling:.1f} dB ceiling)")

        # Rule 2: Nighttime construction / industrial noise ordinance violation
        construction_classes = {"drilling", "jackhammer", "construction"}
        if is_nighttime and sound_class in construction_classes and confidence >= 0.50:
            reasons.append(
                f"Nighttime construction noise violation: '{sound_class}' at {hour:02d}:00 "
                f"(prohibited {self.night_start_hour:02d}:00-{self.night_end_hour:02d}:00)"
            )

        # Rule 3: High-priority emergency sound in sensitive / quiet zone
        emergency_classes = {"gun_shot", "siren", "car_horn"}
        quiet_zones = {"residential", "park", "quiet"}
        if sound_class in emergency_classes and zone_type in quiet_zones and confidence >= 0.55:
            reasons.append(
                f"Alarming acoustic event: '{sound_class}' ({confidence*100:.0f}% conf) in quiet {zone_type} zone"
            )

        # Rule 4: Nighttime sleep disturbance in residential area
        if is_nighttime and zone_type in ("residential", "park") and calibrated_db >= 65.0:
            reasons.append(f"Nighttime residential disturbance ({calibrated_db:.1f} dB exceeds 65 dB quiet limit)")

        return reasons

    def train_isolation_forest(
        self,
        observations: Sequence[Any],
        contamination: float = 0.05,
    ) -> None:
        """Fit an Isolation Forest on acoustic feature vectors."""
        feature_matrix = []
        for obs in observations:
            feat = self._extract_feature_vector(obs)
            if feat is not None:
                feature_matrix.append(feat)

        if len(feature_matrix) < 10:
            # Need minimum observations to train Isolation Forest
            return

        X = np.array(feature_matrix)
        self.ml_model = IsolationForest(
            n_estimators=100,
            contamination=contamination,
            random_state=42,
        )
        self.ml_model.fit(X)
        self._is_ml_fitted = True

    def predict_ml(self, feature_vector: np.ndarray) -> tuple[float, bool]:
        """Predict anomaly score using Isolation Forest.

        Returns:
            (normalized_score [0.0 - 1.0], is_ml_anomaly)
        """
        if not self._is_ml_fitted or self.ml_model is None:
            return 0.0, False

        feat_reshaped = feature_vector.reshape(1, -1)
        # raw decision_function: lower means more anomalous (typically in [-0.5, 0.5])
        raw_score = float(self.ml_model.decision_function(feat_reshaped)[0])
        # Map raw score so that higher score represents higher anomaly likelihood
        normalized_score = round(float(1.0 / (1.0 + np.exp(raw_score * 4.0))), 3)
        prediction = self.ml_model.predict(feat_reshaped)[0]
        is_anomaly = bool(prediction == -1)

        return normalized_score, is_anomaly

    def evaluate_observation(
        self,
        event: Union[dict, Any],
        zone: Optional[UrbanZone] = None,
    ) -> dict:
        """Perform comprehensive multi-factor anomaly evaluation on an observation.

        Args:
            event: Observation dictionary or ORM instance
            zone: Optional pre-resolved UrbanZone

        Returns:
            Dictionary containing anomaly determination, score, severity, and reasons.
        """
        # 1. Extract observation fields
        if isinstance(event, dict):
            lat = float(event.get("latitude", 0.0))
            lon = float(event.get("longitude", 0.0))
            db_val = float(event.get("calibrated_db", event.get("noise_level_db", 0.0)))
            sound_class = str(event.get("sound_class", "unknown"))
            confidence = float(event.get("confidence", 0.5))
            ts = event.get("timestamp")
        else:
            lat = float(getattr(event, "latitude", 0.0))
            lon = float(getattr(event, "longitude", 0.0))
            db_val = float(getattr(event, "calibrated_db", getattr(event, "noise_level_db", 0.0)))
            sound_class = str(getattr(event, "sound_class", "unknown"))
            confidence = float(getattr(event, "confidence", 0.5))
            ts = getattr(event, "timestamp", None)

        hour = self._extract_hour(ts)

        # 2. Resolve urban zone
        resolved_zone = zone or self.zone_registry.find_zone_for_point(lat, lon)
        zone_type = resolved_zone.zone_type if resolved_zone else "general"

        # 3. Statistical Baseline and Z-score
        mean_db, std_db = self.get_baseline(zone_type, hour)
        z_score = self.compute_z_score(db_val, mean_db, std_db)

        # 4. Contextual Rules
        context_reasons = self.check_contextual_rules(
            calibrated_db=db_val,
            sound_class=sound_class,
            confidence=confidence,
            hour=hour,
            zone_type=zone_type,
        )

        # 5. Machine Learning Score (Isolation Forest)
        ml_score = 0.0
        ml_anomaly = False
        feat_vec = self._extract_feature_vector(event, resolved_zone)
        if feat_vec is not None and self._is_ml_fitted:
            ml_score, ml_anomaly = self.predict_ml(feat_vec)
            if ml_anomaly:
                context_reasons.append(f"Isolation Forest flagged anomaly (score: {ml_score:.2f})")

        # 6. Synthesize Severity & Reasons
        reasons = list(context_reasons)
        if z_score >= self.z_critical_threshold:
            reasons.insert(0, f"Critical noise spike (z={z_score:+.1f} sigma, baseline={mean_db:.1f}+/-{std_db:.1f} dB)")
        elif z_score >= self.z_moderate_threshold:
            reasons.insert(0, f"Elevated noise level (z={z_score:+.1f} sigma, baseline={mean_db:.1f}+/-{std_db:.1f} dB)")

        # Calculate composite anomaly score
        # Base is z_score; boost if contextual violations exist
        composite_score = max(z_score, 0.0)
        if context_reasons:
            composite_score = max(composite_score, 2.0) + (0.5 * len(context_reasons))
        composite_score = round(composite_score, 2)

        # Determine severity level
        if (
            z_score >= self.z_critical_threshold
            or db_val >= self.decibel_ceiling
            or "gun_shot" in sound_class
            or composite_score >= 3.5
        ):
            severity = "CRITICAL"
        elif z_score >= self.z_moderate_threshold or len(context_reasons) >= 2 or composite_score >= 2.5:
            severity = "HIGH"
        elif z_score >= 1.5 or len(context_reasons) >= 1 or ml_anomaly:
            severity = "MODERATE"
        else:
            severity = "NORMAL"

        is_anomaly = severity != "NORMAL"

        return {
            "is_anomaly": is_anomaly,
            "anomaly_score": composite_score,
            "severity": severity,
            "reasons": reasons,
            "z_score": z_score,
            "ml_score": ml_score,
            "baseline_mean": mean_db,
            "baseline_std": std_db,
            "zone_type": zone_type,
            "hour_of_day": hour,
        }

    def _extract_hour(self, ts: Any) -> int:
        """Extract hour of day [0 - 23] from datetime object or ISO string."""
        if isinstance(ts, datetime):
            return ts.hour
        if isinstance(ts, str):
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                return dt.hour
            except Exception:
                pass
        return datetime.now(timezone.utc).hour

    def _extract_feature_vector(
        self,
        event: Union[dict, Any],
        zone: Optional[UrbanZone] = None,
    ) -> Optional[np.ndarray]:
        """Extract [calibrated_db, duration, confidence, hour_of_day, zone_code] feature vector."""
        if isinstance(event, dict):
            lat = float(event.get("latitude", 0.0))
            lon = float(event.get("longitude", 0.0))
            db_val = float(event.get("calibrated_db", event.get("noise_level_db", 0.0)))
            confidence = float(event.get("confidence", 0.5))
            duration = float(event.get("duration", 4.0))
            ts = event.get("timestamp")
        else:
            lat = float(getattr(event, "latitude", 0.0))
            lon = float(getattr(event, "longitude", 0.0))
            db_val = float(getattr(event, "calibrated_db", getattr(event, "noise_level_db", 0.0)))
            confidence = float(getattr(event, "confidence", 0.5))
            duration = float(getattr(event, "duration", 4.0))
            ts = getattr(event, "timestamp", None)

        hour = self._extract_hour(ts)
        resolved_zone = zone or self.zone_registry.find_zone_for_point(lat, lon)
        z_type = resolved_zone.zone_type if resolved_zone else "general"
        z_code = ZONE_TYPE_CODES.get(z_type, 3)

        return np.array([db_val, duration, confidence, float(hour), float(z_code)], dtype=np.float32)
