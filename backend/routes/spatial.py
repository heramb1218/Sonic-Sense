"""
Spatial Intelligence & Anomaly REST API Routes

Endpoints:
- GET /heatmap: Normalized spatial intensity points for frontend map heatmap rendering
- GET /hexagons: Aggregated spatial hexagon data (H3 layer)
- GET /anomalies: Acoustic observations flagged as anomalous
"""

from typing import Optional
from fastapi import APIRouter, Depends, Query

from src.database.connection import get_db_manager
from src.database.repository import ObservationRepository
from backend.schemas.observation import HeatmapPoint, ObservationResponse

router = APIRouter(tags=["Spatial Intelligence"])


def get_repository() -> ObservationRepository:
    return ObservationRepository(db_manager=get_db_manager())


@router.get("/heatmap", response_model=list[HeatmapPoint])
def get_heatmap_points(
    limit: int = Query(300, ge=1, le=1000, description="Max points for heatmap"),
    repo: ObservationRepository = Depends(get_repository),
):
    """Retrieve spatial observation points with normalized decibel intensity for heatmaps.

    Intensity is scaled linearly between 30 dB (0.05 min intensity) and 100 dB (1.0 max intensity).
    """
    records = repo.get_latest_observations(limit=limit)
    points = []

    for r in records:
        # Scale 30dB -> 100dB to [0.05, 1.0]
        db_val = float(r.calibrated_db)
        intensity = max(0.05, min(1.0, (db_val - 30.0) / 70.0))
        points.append(HeatmapPoint(
            latitude=r.latitude,
            longitude=r.longitude,
            intensity=round(intensity, 3),
            calibrated_db=r.calibrated_db,
            sound_class=r.sound_class,
        ))

    return points


@router.get("/hexagons")
def get_hexagonal_aggregations(
    resolution: int = Query(8, ge=5, le=11, description="H3 index resolution"),
    repo: ObservationRepository = Depends(get_repository),
):
    """Retrieve hexagonal cell-level acoustic aggregations (Uber H3).

    Note: In Phase 7, returns cell-mapped observations; fully expanded in Phase 8.
    """
    records = repo.get_latest_observations(limit=500)
    cells = {}

    for r in records:
        cell_id = r.h3_cell or f"cell_{round(r.latitude, 3)}_{round(r.longitude, 3)}"
        if cell_id not in cells:
            cells[cell_id] = {
                "h3_cell": cell_id,
                "center_lat": r.latitude,
                "center_lon": r.longitude,
                "observation_count": 0,
                "noise_levels": [],
                "sound_classes": {},
            }

        c = cells[cell_id]
        c["observation_count"] += 1
        c["noise_levels"].append(r.calibrated_db)
        c["sound_classes"][r.sound_class] = c["sound_classes"].get(r.sound_class, 0) + 1

    # Compute per-cell statistics
    result = []
    for cell_id, data in cells.items():
        levels = data["noise_levels"]
        dominant_sound = max(data["sound_classes"].items(), key=lambda x: x[1])[0]
        result.append({
            "h3_cell": cell_id,
            "center_lat": data["center_lat"],
            "center_lon": data["center_lon"],
            "observation_count": data["observation_count"],
            "average_db": round(sum(levels) / len(levels), 1),
            "max_db": round(max(levels), 1),
            "dominant_sound_class": dominant_sound,
        })

    return result


@router.get("/anomalies", response_model=list[ObservationResponse])
def get_anomalies(
    min_score: float = Query(2.0, ge=0.5, le=10.0, description="Minimum anomaly z-score"),
    limit: int = Query(50, ge=1, le=200),
    repo: ObservationRepository = Depends(get_repository),
):
    """Fetch acoustic observations that deviate significantly from baseline."""
    anomalies = repo.get_anomalies(min_score=min_score, limit=limit)
    return [r.to_dict() for r in anomalies]
