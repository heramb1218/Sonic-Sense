"""
Pydantic Schemas for Backend REST API and WebSockets

Provides strict validation and serialization for observation payloads,
spatial queries, heatmap points, and analytical summaries.
"""

import uuid
from datetime import datetime, timezone
from typing import Optional, Union, Any
from pydantic import BaseModel, Field, ConfigDict, model_validator


class ObservationCreate(BaseModel):
    """Schema for validating incoming acoustic observation payloads (POST /observations)."""
    model_config = ConfigDict(extra="ignore")

    event_id: Optional[str] = Field(None, max_length=50, description="Unique observation event identifier")
    device_id: str = Field(..., min_length=1, max_length=50, description="Unique sensing node identifier")
    device_model: Optional[str] = Field(None, max_length=100, description="Hardware model name (e.g. Google Pixel 7)")
    timestamp: Optional[Union[datetime, str]] = Field(None, description="ISO-8601 observation timestamp")
    latitude: float = Field(..., ge=-90.0, le=90.0, description="WGS-84 Latitude in decimal degrees (-90 to 90)")
    longitude: float = Field(..., ge=-180.0, le=180.0, description="WGS-84 Longitude in decimal degrees (-180 to 180)")
    gps_accuracy: Optional[float] = Field(None, ge=0.0, le=1000.0, description="GPS horizontal accuracy radius in meters")
    noise_level_db: float = Field(..., ge=0.0, le=150.0, description="Raw uncalibrated sound pressure level in dB SPL (0 to 150)")
    sound_class: str = Field(..., min_length=1, max_length=50, description="Acoustic sound category")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Classification confidence probability [0.0 - 1.0]")
    calibration_offset: float = Field(0.0, ge=-30.0, le=30.0, description="Microphone hardware calibration offset in dB")
    calibrated_db: Optional[float] = Field(None, ge=0.0, le=150.0, description="Calibrated sound pressure level in dB SPL")
    h3_cell: Optional[str] = Field(None, max_length=20, description="Uber H3 hexagonal grid cell index")
    anomaly_score: Optional[float] = Field(None, description="Statistical or ML anomaly deviation score")
    anomaly_severity: Optional[str] = Field(None, max_length=20, description="Anomaly severity tier")
    anomaly_reasons: Optional[list[str]] = Field(default_factory=list, description="Descriptive list of anomaly trigger reasons")
    audio_file: Optional[str] = Field(None, max_length=255, description="Audio file reference or sample name")

    @model_validator(mode="after")
    def compute_defaults(self) -> "ObservationCreate":
        if not self.event_id:
            self.event_id = str(uuid.uuid4())
        if self.calibrated_db is None:
            self.calibrated_db = round(self.noise_level_db + (self.calibration_offset or 0.0), 1)
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc).isoformat()
        elif isinstance(self.timestamp, datetime):
            self.timestamp = self.timestamp.isoformat()
        return self


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
    anomaly_severity: Optional[str] = None
    anomaly_reasons: Optional[list[str]] = None
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
