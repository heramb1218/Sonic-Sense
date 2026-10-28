"""
Statistics and Metrics REST API Routes

Endpoints:
- GET /statistics: Overall platform summary (counts, class distribution, baseline)
- GET /sound-classes: Predefined UrbanSound8K taxonomy classes
- GET /statistics/area: Area-level acoustic aggregates within bounding box
"""

from fastapi import APIRouter, Depends, Query

from src.config.settings import settings
from src.database.repository import ObservationRepository
from backend.dependencies import get_repository
from backend.schemas.observation import (
    GlobalStatisticsResponse,
    SoundClassesResponse,
    AreaStatisticsResponse,
)

router = APIRouter(tags=["Statistics & Analytics"])


@router.get("/statistics", response_model=GlobalStatisticsResponse)
def get_global_statistics(repo: ObservationRepository = Depends(get_repository)):
    """Retrieve system-wide summary metrics for dashboard analytics cards."""
    counts = repo.get_sound_event_counts()
    baseline = repo.get_historical_baseline(time_window_hours=24)

    total_obs = sum(counts.values())
    latest = repo.get_latest_observations(limit=50)
    unique_devices = len({r.device_id for r in latest})

    return GlobalStatisticsResponse(
        total_observations=total_obs,
        sound_class_distribution=counts,
        baseline_mean_db=baseline["mean_db"],
        baseline_std_db=baseline["std_db"],
        active_devices_count=unique_devices,
    )


@router.get("/sound-classes", response_model=SoundClassesResponse)
def get_sound_classes():
    """Retrieve official UrbanSound8K taxonomy class mapping."""
    return SoundClassesResponse(
        classes=settings.URBAN_SOUND_CLASSES,
        total_classes=settings.NUM_CLASSES,
    )


@router.get("/statistics/area", response_model=AreaStatisticsResponse)
def get_area_statistics(
    min_lat: float = Query(..., description="South boundary latitude"),
    min_lon: float = Query(..., description="West boundary longitude"),
    max_lat: float = Query(..., description="North boundary latitude"),
    max_lon: float = Query(..., description="East boundary longitude"),
    repo: ObservationRepository = Depends(get_repository),
):
    """Calculate acoustic aggregates (average dB, min, max, count) inside a geographic bounding box."""
    stats = repo.get_average_noise_by_area(min_lat, min_lon, max_lat, max_lon)
    return AreaStatisticsResponse(
        average_db=stats["average_db"],
        max_db=stats["max_db"],
        min_db=stats["min_db"],
        count=stats["count"],
    )
