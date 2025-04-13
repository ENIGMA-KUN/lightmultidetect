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
        # Ensure upload directory exists
        os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
        
        # Ensure model weights directory exists
        os.makedirs(settings.MODEL_WEIGHTS_DIR, exist_ok=True)
        
        # Preload models if not in debug mode
        if not settings.DEBUG:
            from app.models.image_models import get_image_model
            from app.models.audio_models import get_audio_model
            from app.models.video_models import get_video_model
            
            logger.info("Preloading ML models...")
            _ = get_image_model()
            _ = get_audio_model()
            _ = get_video_model()
            logger.info("ML models loaded successfully.")
    
    return start_app


def create_stop_app_handler(app: FastAPI) -> Callable:
    """
    Function to handle shutdown events
    """
    async def stop_app() -> None:
        logger.info("Shutting down application...")
    
    return stop_app