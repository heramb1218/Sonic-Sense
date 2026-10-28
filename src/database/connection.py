"""
Database Connection Manager — Engine, Sessions, and Auto-Provisioning

Supports:
- PostgreSQL + PostGIS when available
- Local SQLite database fallback (urban_noise.db) for zero-dependency local development
- Transactional context manager (session_scope)
"""

import logging
from contextlib import contextmanager
from typing import Optional, Generator
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session

from src.config.settings import settings
from src.database.models import Base

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages database connection lifecycle and schema setup."""

    def __init__(self, db_url: Optional[str] = None, echo: bool = False):
        self.db_url = db_url or self._resolve_db_url()
        self.echo = echo

        connect_args = {}
        engine_kwargs = {}
        if self.db_url.startswith("sqlite"):
            connect_args["check_same_thread"] = False
        if self.db_url == "sqlite:///:memory:":
            from sqlalchemy.pool import StaticPool
            engine_kwargs["poolclass"] = StaticPool

        self.engine = create_engine(
            self.db_url,
            echo=self.echo,
            connect_args=connect_args,
            **engine_kwargs,
        )
        self.SessionFactory = sessionmaker(bind=self.engine, autocommit=False, autoflush=False)
        self.init_db()

    def _resolve_db_url(self) -> str:
        """Attempt PostgreSQL connection first, fallback cleanly to SQLite."""
        pg_url = settings.POSTGRES_DATABASE_URL
        try:
            test_engine = create_engine(pg_url, connect_args={"connect_timeout": 1})
            with test_engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("Connected to PostgreSQL + PostGIS database.")
            return pg_url
        except Exception:
            logger.info(f"PostgreSQL not available. Using local SQLite storage at: {settings.DATABASE_URL}")
            return settings.DATABASE_URL

    def init_db(self) -> None:
        """Create all registered database tables and indexes."""
        Base.metadata.create_all(self.engine)
        logger.debug("Database schema and indexes initialized.")

    @contextmanager
    def session_scope(self) -> Generator[Session, None, None]:
        """Provide a transactional scope around a series of operations."""
        session = self.SessionFactory()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session rolled back due to error: {e}")
            raise
        finally:
            session.close()

    def drop_all(self) -> None:
        """Drop all tables (used for testing resets)."""
        Base.metadata.drop_all(self.engine)


# Singleton instance
_db_manager: Optional[DatabaseManager] = None


def get_db_manager(db_url: Optional[str] = None) -> DatabaseManager:
    """Retrieve or initialize singleton database manager."""
    global _db_manager
    if _db_manager is None or db_url is not None:
        _db_manager = DatabaseManager(db_url=db_url)
    return _db_manager


def reset_db_manager() -> None:
    """Reset the singleton database manager instance."""
    global _db_manager
    if _db_manager is not None:
        try:
            _db_manager.engine.dispose()
        except Exception:
            pass
        _db_manager = None
