"""
Observations REST API Routes

Endpoints:
- GET /observations: Query observations with optional time, class, and spatial bounding filters
- GET /observations/latest: Retrieve the most recent N observations
- POST /observations: Ingest a new observation, save to DB, and broadcast to WebSockets
"""

from datetime import datetime
from typing import Optional
from fastapi import APIRouter, Depends, Query, HTTPException, status

from src.database.repository import ObservationRepository
from backend.dependencies import get_repository
from backend.schemas.observation import ObservationCreate, ObservationResponse
from backend.websocket.connection_manager import ws_manager

router = APIRouter(prefix="/observations", tags=["Observations"])


@router.get("", response_model=list[ObservationResponse])
def get_observations(
    start_time: Optional[datetime] = Query(None, description="Start timestamp (ISO-8601)"),
    end_time: Optional[datetime] = Query(None, description="End timestamp (ISO-8601)"),
    sound_class: Optional[str] = Query(None, description="Filter by sound category (e.g. siren)"),
    min_lat: Optional[float] = Query(None, description="Bounding box minimum latitude"),
    min_lon: Optional[float] = Query(None, description="Bounding box minimum longitude"),
    max_lat: Optional[float] = Query(None, description="Bounding box maximum latitude"),
    max_lon: Optional[float] = Query(None, description="Bounding box maximum longitude"),
    limit: int = Query(100, ge=1, le=1000, description="Max observations to return"),
    repo: ObservationRepository = Depends(get_repository),
):
    """Retrieve observations supporting spatial bounding box, time-series, or class filters."""
    # 1. Bounding box filter
    if all(v is not None for v in (min_lat, min_lon, max_lat, max_lon)):
        records = repo.get_observations_in_bounding_box(min_lat, min_lon, max_lat, max_lon, limit=limit)
    # 2. Time range filter
    elif start_time is not None and end_time is not None:
        records = repo.get_observations_in_time_range(start_time, end_time, limit=limit)
    # 3. Class filter
    elif sound_class is not None:
        records = repo.get_observations_by_class(sound_class, limit=limit)
    # 4. Default: latest observations
    else:
        records = repo.get_latest_observations(limit=limit)

    return [r.to_dict() for r in records]


@router.get("/latest", response_model=list[ObservationResponse])
def get_latest_observations(
    limit: int = Query(50, ge=1, le=500, description="Number of recent records"),
    repo: ObservationRepository = Depends(get_repository),
):
    """Fetch the most recent observations sorted chronologically descending."""
    records = repo.get_latest_observations(limit=limit)
    return [r.to_dict() for r in records]


@router.post("", response_model=ObservationResponse, status_code=status.HTTP_201_CREATED)
async def create_observation(
    event: ObservationCreate,
    repo: ObservationRepository = Depends(get_repository),
):
    """Ingest a new observation, persist to database, and broadcast via WebSocket."""
    try:
        event_dict = event.model_dump()
        obs = repo.insert_observation(event_dict)
        response_dict = obs.to_dict()

        # Attach real-time anomaly severity and textual reasons for live frontend alerts
        if event.anomaly_severity:
            response_dict["anomaly_severity"] = event.anomaly_severity
            response_dict["anomaly_reasons"] = event.anomaly_reasons or []
        elif obs.anomaly_score is not None:
            eval_res = repo.anomaly_detector.evaluate_observation(obs)
            response_dict["anomaly_severity"] = eval_res["severity"]
            response_dict["anomaly_reasons"] = eval_res["reasons"]

        # Broadcast asynchronously to all connected dashboard WebSocket clients
        await ws_manager.broadcast_event(response_dict)
        return response_dict
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to record observation: {e}",
        )
