"""
FastAPI Dependency Providers for Backend Services

Provides reusable, injectable dependencies for database repositories and services.
These dependencies can be overridden during testing via app.dependency_overrides.
"""

from src.database.connection import get_db_manager
from src.database.repository import ObservationRepository


def get_repository() -> ObservationRepository:
    """Dependency injection provider for ObservationRepository."""
    return ObservationRepository(db_manager=get_db_manager())

