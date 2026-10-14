"""
Unit and Integration Tests for Acoustic Anomaly Detection Engine (Phase 9)

Tests:
- Day vs night spatial baseline decibel retrieval
- Statistical z-score computation and standard deviation safety
- Contextual rule evaluation (90 dB ceiling, nighttime construction, quiet zone sirens/gunshots)
- Unsupervised Isolation Forest fitting and inference
- Composite multi-tier severity classification (NORMAL, MODERATE, HIGH, CRITICAL)
- Empirical baseline adaptation from observation history
- Integration with streaming consumer (automatic anomaly scoring and alert emissions)
"""

from datetime import datetime, timezone
import pytest
import numpy as np

from src.anomaly.detector import AcousticAnomalyDetector, DEFAULT_ZONE_BASELINES
from src.streaming.consumer import NoiseEventConsumer
from src.streaming.broker import LocalQueueBroker, StreamMessage


@pytest.fixture
def detector():
    """Create a fresh AcousticAnomalyDetector instance."""
    return AcousticAnomalyDetector()


class TestAcousticBaselinesAndZScores:
    """Tests for statistical baseline management and z-score calculations."""

    def test_baseline_day_and_night(self, detector):
        """Test that daytime and nighttime baselines differ appropriately."""
        res_day_mean, res_day_std = detector.get_baseline("residential", hour=14)
        res_night_mean, res_night_std = detector.get_baseline("residential", hour=2)

        # Residential night is quieter than day
        assert res_night_mean < res_day_mean
        assert res_day_mean == 52.0
        assert res_night_mean == 42.0

        # Traffic is significantly louder than residential
        traf_day_mean, _ = detector.get_baseline("traffic", hour=14)
        assert traf_day_mean > res_day_mean

    def test_compute_z_score(self, detector):
        """Test z-score mathematical accuracy."""
        # Value exactly at mean -> z = 0.0
        assert detector.compute_z_score(calibrated_db=50.0, mean_db=50.0, std_db=5.0) == 0.0

        # +2 standard deviations -> z = 2.0
        assert detector.compute_z_score(calibrated_db=60.0, mean_db=50.0, std_db=5.0) == 2.0

        # -1 standard deviation -> z = -1.0
        assert detector.compute_z_score(calibrated_db=45.0, mean_db=50.0, std_db=5.0) == -1.0

        # Zero std guard: minimum effective std is 1.0, avoids ZeroDivisionError
        z_safe = detector.compute_z_score(calibrated_db=55.0, mean_db=50.0, std_db=0.0)
        assert z_safe == 5.0


class TestContextualRules:
    """Tests for domain-specific urban noise regulation rules."""

    def test_decibel_ceiling_violation(self, detector):
        """Test that sound >= 90 dB triggers extreme spike violation."""
        reasons = detector.check_contextual_rules(
            calibrated_db=92.5,
            sound_class="car_horn",
            confidence=0.85,
            hour=14,
            zone_type="commercial",
        )
        assert any("Extreme acoustic spike" in r for r in reasons)

    def test_nighttime_construction_violation(self, detector):
        """Test that nighttime drilling triggers municipal violation, while daytime does not."""
        # 23:00 (11 PM) -> nighttime violation
        night_reasons = detector.check_contextual_rules(
            calibrated_db=75.0,
            sound_class="drilling",
            confidence=0.80,
            hour=23,
            zone_type="residential",
        )
        assert any("Nighttime construction noise violation" in r for r in night_reasons)

        # 14:00 (2 PM) -> permitted construction hour
        day_reasons = detector.check_contextual_rules(
            calibrated_db=75.0,
            sound_class="drilling",
            confidence=0.80,
            hour=14,
            zone_type="construction",
        )
        assert not any("Nighttime construction noise violation" in r for r in day_reasons)

    def test_alarming_sound_in_quiet_zone(self, detector):
        """Test that sirens or gunshots in residential/park zones trigger an alert."""
        reasons = detector.check_contextual_rules(
            calibrated_db=72.0,
            sound_class="siren",
            confidence=0.88,
            hour=15,
            zone_type="park",
        )
        assert any("Alarming acoustic event" in r for r in reasons)


class TestUnsupervisedMLAndFullEvaluation:
    """Tests for Isolation Forest and multi-tier evaluation."""

    def test_isolation_forest_train_and_predict(self, detector):
        """Test fitting Isolation Forest and computing anomaly score."""
        # Generate synthetic normal feature vectors
        np.random.seed(42)
        normal_samples = []
        for _ in range(50):
            normal_samples.append({
                "latitude": 40.71,
                "longitude": -74.00,
                "calibrated_db": float(np.random.normal(55.0, 4.0)),
                "confidence": 0.85,
                "duration": 4.0,
                "timestamp": "2026-10-14T14:00:00Z",
            })

        detector.train_isolation_forest(normal_samples, contamination=0.1)
        assert detector._is_ml_fitted is True

        # Test normal point
        normal_feat = np.array([55.0, 4.0, 0.85, 14.0, 1.0], dtype=np.float32)
        score_normal, is_anom_normal = detector.predict_ml(normal_feat)
        assert 0.0 <= score_normal <= 1.0

        # Test extreme outlier point (110 dB at 03:00 AM)
        outlier_feat = np.array([110.0, 4.0, 0.99, 3.0, 0.0], dtype=np.float32)
        score_outlier, is_anom_outlier = detector.predict_ml(outlier_feat)
        assert score_outlier > score_normal

    def test_evaluate_observation_normal_event(self, detector):
        """Test evaluation of an ordinary, expected acoustic event."""
        normal_event = {
            "latitude": 40.7128,
            "longitude": -74.0060,
            "calibrated_db": 53.0,
            "sound_class": "street_music",
            "confidence": 0.80,
            "timestamp": "2026-10-14T14:00:00Z",  # 2 PM in commercial plaza
        }
        res = detector.evaluate_observation(normal_event)
        assert res["severity"] == "NORMAL"
        assert res["is_anomaly"] is False
        assert res["anomaly_score"] < 2.0

    def test_evaluate_observation_critical_anomaly(self, detector):
        """Test evaluation of a severe acoustic event."""
        critical_event = {
            "latitude": 18.5024,
            "longitude": 73.8388,  # Botanical Park / Residential
            "calibrated_db": 96.0,
            "sound_class": "siren",
            "confidence": 0.95,
            "timestamp": "2026-10-14T02:30:00Z",  # 2:30 AM
        }
        res = detector.evaluate_observation(critical_event)
        assert res["severity"] == "CRITICAL"
        assert res["is_anomaly"] is True
        assert res["anomaly_score"] >= 3.0
        assert len(res["reasons"]) >= 2

    def test_stream_consumer_populates_anomaly_score(self, detector):
        """Test that NoiseEventConsumer automatically attaches anomaly metadata to streamed events."""
        broker = LocalQueueBroker()
        topic = "test-anomaly-stream"

        raw_event = {
            "event_id": "STREAM-TEST-001",
            "device_id": "DEV-01",
            "latitude": 18.5024,
            "longitude": 73.8388,
            "calibrated_db": 95.0,
            "noise_level_db": 95.0,
            "sound_class": "siren",
            "confidence": 0.92,
            "timestamp": "2026-10-14T01:00:00Z",
            "anomaly_score": None,
        }

        broker.publish(topic, raw_event)
        consumer = NoiseEventConsumer(broker=broker, topic=topic, anomaly_detector=detector)

        consumed_events = list(consumer.consume_events(max_messages=1, timeout=0.5, verbose=False))
        assert len(consumed_events) == 1
        ev = consumed_events[0]

        assert ev["anomaly_score"] is not None
        assert ev["anomaly_score"] >= 3.0
        assert ev["anomaly_severity"] == "CRITICAL"
        assert len(ev["anomaly_reasons"]) > 0
