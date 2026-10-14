"""
Database Models — Spatial-Temporal Noise Observations Table

Defines the SQLAlchemy ORM schema for acoustic observations conforming
to Master Prompt Section 19.

Indexed fields:
- timestamp: for time-series range queries and chronological sorting
- (latitude, longitude): compound spatial index for geographic bounds & radius queries
- sound_class: for category-specific queries and aggregations
- h3_cell: for rapid spatial index lookups
"""

import uuid
from datetime import datetime, timezone
from typing import Optional
from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    DateTime,
    Index,
    Text,
)
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class NoiseObservation(Base):
    """Stores spatial-temporal acoustic sensor observations."""

    __tablename__ = "noise_observations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(String(36), unique=True, nullable=False, index=True)
    device_id = Column(String(50), nullable=False, index=True)
    device_model = Column(String(100), nullable=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)

    # Spatial coordinates
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    gps_accuracy = Column(Float, nullable=True)

    # Acoustic measurements
    noise_level_db = Column(Float, nullable=False)
    sound_class = Column(String(50), nullable=False, index=True)
    confidence = Column(Float, nullable=False)

    # Hardware calibration
    calibration_offset = Column(Float, nullable=False, default=0.0)
    calibrated_db = Column(Float, nullable=False, index=True)

    # Spatial intelligence & anomaly detection placeholders
    h3_cell = Column(String(20), nullable=True, index=True)
    anomaly_score = Column(Float, nullable=True)

    # Metadata
    audio_file = Column(String(255), nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    # Compound spatial and temporal indexes for high query performance
    __table_args__ = (
        Index("idx_spatial_coords", "latitude", "longitude"),
        Index("idx_temporal_class", "timestamp", "sound_class"),
        Index("idx_h3_timestamp", "h3_cell", "timestamp"),
    )

    def to_dict(self) -> dict:
        """Convert ORM record to standardized Section 16 dictionary."""
        return {
            "id": self.id,
            "event_id": self.event_id,
            "device_id": self.device_id,
            "device_model": self.device_model,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "gps_accuracy": self.gps_accuracy,
            "audio_file": self.audio_file,
            "noise_level_db": self.noise_level_db,
            "sound_class": self.sound_class,
            "confidence": self.confidence,
            "calibration_offset": self.calibration_offset,
            "calibrated_db": self.calibrated_db,
            "h3_cell": self.h3_cell,
            "anomaly_score": self.anomaly_score,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    def __repr__(self) -> str:
        return (
            f"<NoiseObservation(id={self.id}, device={self.device_id}, "
            f"class={self.sound_class}, db={self.calibrated_db}, "
            f"loc=[{self.latitude:.4f}, {self.longitude:.4f}])>"
        )
