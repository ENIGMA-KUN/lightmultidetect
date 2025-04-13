from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from typing import Generator
import logging

from app.core.config import settings

logger = logging.getLogger(__name__)

# Create database engine with connection pooling
engine = create_engine(
    settings.SQLALCHEMY_DATABASE_URI,
    pool_pre_ping=True,  # Check connection before using it
    pool_recycle=3600,   # Recycle connections every hour
    connect_args={"options": "-c timezone=utc"}  # Set timezone to UTC
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base class for models
Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """
    Get a database session from the connection pool.
    
    Yields:
        Session: Database session
    
    Notes:
        This function is intended to be used as a FastAPI dependency.
        The session is automatically closed after the request is processed.
    """
    db = SessionLocal()
    try:
        logger.debug("Database session created")
        yield db
    except Exception as e:
        logger.error(f"Database session error: {str(e)}")
        db.rollback()
        raise
    finally:
        logger.debug("Database session closed")
        db.close()


def init_db() -> None:
    """
    Initialize the database by creating all tables.
    
    This function is called during application startup.
    """
    try:
        # Import all models here to ensure they're registered with Base
        from app.db.models import DetectionResult
        
        # Create tables
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        raise