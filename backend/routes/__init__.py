"""Backend Routes Package."""

from backend.routes.observations import router as observations_router
from backend.routes.spatial import router as spatial_router
from backend.routes.statistics import router as statistics_router

__all__ = ["observations_router", "spatial_router", "statistics_router"]
