"""
Pydantic Schemas for Backend REST API and WebSockets

Provides strict validation and serialization for observation payloads,
spatial queries, heatmap points, and analytical summaries.
"""

from datetime import datetime
from typing import Optional, Any
from pydantic import BaseModel, Field, ConfigDict


class ObservationResponse(BaseModel):
    """Schema representing an individual acoustic observation event."""
    model_config = ConfigDict(from_attributes=True)

    id: Optional[int] = None
    event_id: str
    device_id: str
    device_model: Optional[str] = None
    timestamp: str
    latitude: float
    longitude: float
    gps_accuracy: Optional[float] = None
    noise_level_db: float
    sound_class: str
    confidence: float
    calibration_offset: float
    calibrated_db: float
    h3_cell: Optional[str] = None
    anomaly_score: Optional[float] = None
    audio_file: Optional[str] = None


class HeatmapPoint(BaseModel):
    """Weighted spatial point for rendering map heatmaps."""
    latitude: float
    longitude: float
    intensity: float = Field(..., description="Normalized noise intensity [0.0 - 1.0]")
    calibrated_db: float
    sound_class: str


class SoundClassesResponse(BaseModel):
    """Dictionary mapping class IDs to class names."""
    classes: dict[int, str]
    total_classes: int


class AreaStatisticsResponse(BaseModel):
    """Acoustic statistics aggregated over a spatial region."""
    average_db: float
    max_db: float
    min_db: float
    count: int


class GlobalStatisticsResponse(BaseModel):
    """System-wide summary metrics."""
    total_observations: int
    sound_class_distribution: dict[str, int]
    baseline_mean_db: float
    baseline_std_db: float
    active_devices_count: int


class HealthResponse(BaseModel):
    """API health status and environment summary."""
    status: str
    timestamp: str
    database_engine: str
    streaming_mode: str
    active_connections: int
