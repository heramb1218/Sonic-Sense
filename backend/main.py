"""
FastAPI Main Application — Real-Time Acoustic Intelligence Backend

Responsibilities:
- Expose REST API endpoints for observations, heatmaps, H3 aggregations, and statistics
- Manage real-time WebSocket connection hub (/ws/live) for live dashboard map streaming
- Enable Cross-Origin Resource Sharing (CORS) for local frontend dashboards
- Connect database layer and initialize tables on startup

Usage:
    # Run dev server with auto-reload:
    uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
"""

import sys
import logging
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from contextlib import asynccontextmanager
import uvicorn
from fastapi import FastAPI, Depends, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from src.config.settings import settings
from src.database.connection import get_db_manager
from src.database.repository import ObservationRepository
from backend.dependencies import get_repository
from backend.routes.observations import router as observations_router
from backend.routes.spatial import router as spatial_router
from backend.routes.statistics import router as statistics_router
from backend.websocket.connection_manager import ws_manager
from backend.schemas.observation import HealthResponse

# Configure logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ensure database schema is provisioned on application launch."""
    logger.info("Initializing database connection on startup...")
    db = get_db_manager()
    db.init_db()
    logger.info("FastAPI backend ready.")
    yield


# Initialize FastAPI application
app = FastAPI(
    title="Urban Noise Intelligence Platform API",
    description=(
        "Real-Time Spatial-Temporal Acoustic Intelligence Platform for Urban "
        "Noise Monitoring, Sound Classification, and Anomaly Detection."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for frontend web application
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register route modules
app.include_router(observations_router)
app.include_router(spatial_router)
app.include_router(statistics_router)


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check(repo: ObservationRepository = Depends(get_repository)):
    """System health check and environmental status."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(timezone.utc).isoformat(),
        database_engine=repo.db.engine.dialect.name.upper(),
        streaming_mode=settings.STREAMING_MODE,
        active_connections=ws_manager.connection_count,
    )


@app.websocket("/ws/live")
async def websocket_live_stream(websocket: WebSocket):
    """Real-time WebSocket endpoint streaming acoustic observation events to the map dashboard."""
    await ws_manager.connect(websocket)
    try:
        while True:
            # Keep connection open and receive optional ping messages
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception as e:
        logger.warning(f"WebSocket client error: {e}")
        ws_manager.disconnect(websocket)


# Mount frontend static web application (catch-all at root)
FRONTEND_DIR = PROJECT_ROOT / "frontend"
if FRONTEND_DIR.exists():
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


if __name__ == "__main__":
    uvicorn.run(
        "backend.main:app",
        host=settings.BACKEND_HOST,
        port=settings.BACKEND_PORT,
        reload=False,
    )
