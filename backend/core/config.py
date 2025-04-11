import os
import secrets
from typing import List, Optional, Union
from pydantic import BaseSettings, validator


class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str = secrets.token_urlsafe(32)
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8  # 8 days
    
    # CORS settings
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8000"]
    
    # Directory settings
    UPLOAD_DIR: str = "data/uploads"
    RESULTS_DIR: str = "data/results"
    MODEL_DIR: str = "ml/models/weights"
    
    # Cleanup settings
    CLEANUP_ON_SHUTDOWN: bool = True
    
    # Redis settings
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    
    # Celery settings
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/0"
    
    # Model settings
    DEFAULT_MODEL: str = "lightweight_multimodal"
    AVAILABLE_MODELS: List[str] = ["lightweight_multimodal", "image_only", "video_only"]
    
    # Security settings
    ALGORITHM: str = "HS256"
    
    class Config:
        env_file = ".env"


settings = Settings() 