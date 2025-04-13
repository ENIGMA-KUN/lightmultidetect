import os
import logging
from typing import Callable
from fastapi import FastAPI
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine

from app.core.config import settings
from app.db.session import get_db, engine

Base = declarative_base()

logger = logging.getLogger(__name__)


def create_start_app_handler(app: FastAPI) -> Callable:
    """
    Function to handle startup events
    """
    async def start_app() -> None:
        # Ensure required directories exist
        os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
        os.makedirs(settings.MODEL_WEIGHTS_DIR, exist_ok=True)
        os.makedirs("visualizations", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        
        logger.info(f"Starting application in {settings.ENVIRONMENT} environment")
        
        # Initialize database tables
        from app.db.models import Base
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables initialized")
        
        # Initialize preprocessing modules
        try:
            logger.info("Initializing preprocessing modules...")
            from app.preprocessing import image_preprocessing, audio_preprocessing, video_preprocessing
            logger.info("Preprocessing modules initialized successfully")
        except ImportError as e:
            logger.warning(f"Error initializing preprocessing modules: {str(e)}")
        
        # Preload models if not in debug mode
        if not settings.DEBUG:
            try:
                from app.models.image_models import get_image_model
                from app.models.audio_models import get_audio_model
                from app.models.video_models import get_video_model
                
                logger.info("Preloading ML models...")
                _ = get_image_model()
                _ = get_audio_model()
                _ = get_video_model()
                logger.info("ML models loaded successfully")
            except Exception as e:
                logger.error(f"Error preloading ML models: {str(e)}")
        else:
            logger.info("Debug mode: Skipping model preloading")
    
    return start_app


def create_stop_app_handler(app: FastAPI) -> Callable:
    """
    Function to handle shutdown events
    """
    async def stop_app() -> None:
        logger.info("Shutting down application...")
        
        # Clean up any resources
        try:
            # Close any open model instances or resources
            import gc
            gc.collect()
            logger.info("Resources cleaned up successfully")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
    
    return stop_app