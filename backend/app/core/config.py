import os
from dotenv import load_dotenv
from pydantic import BaseSettings, Field, validator
from typing import List, Optional, Union, Dict, Any

# Load environment variables from .env file
load_dotenv()

class Settings(BaseSettings):
    # Environment
    ENVIRONMENT: str = Field(default="development")
    DEBUG: bool = Field(default=False)
    SECRET_KEY: str = Field(default="")
    
    # API settings
    API_V1_STR: str = Field(default="/api/v1")
    PROJECT_NAME: str = Field(default="DeepFake Detection Platform")
    
    # CORS settings
    BACKEND_CORS_ORIGINS: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8000"]
    )
    
    # Database settings
    POSTGRES_SERVER: str = Field(default="db")
    POSTGRES_USER: str = Field(default="postgres")
    POSTGRES_PASSWORD: str = Field(default="postgres")
    POSTGRES_DB: str = Field(default="deepfake_detection")
    POSTGRES_PORT: int = Field(default=5432)
    SQLALCHEMY_DATABASE_URI: Optional[str] = Field(default=None)
    
    # Redis settings
    REDIS_HOST: str = Field(default="redis")
    REDIS_PORT: int = Field(default=6379)
    REDIS_URL: str = Field(default="redis://redis:6379/0")
    
    # Celery settings
    CELERY_BROKER_URL: str = Field(default="redis://redis:6379/0")
    CELERY_RESULT_BACKEND: str = Field(default="redis://redis:6379/0")
    CELERY_TASK_ALWAYS_EAGER: bool = Field(default=False)
    
    # ML Model settings
    MODEL_WEIGHTS_DIR: str = Field(default="app/models/weights")
    IMAGE_MODEL_TYPE: str = Field(default="xception")
    AUDIO_MODEL_TYPE: str = Field(default="wav2vec2")
    VIDEO_MODEL_TYPE: str = Field(default="3dcnn")
    
    # File upload settings
    UPLOAD_DIR: str = Field(default="uploads")
    MAX_UPLOAD_SIZE: int = Field(default=100 * 1024 * 1024)  # 100MB
    ALLOWED_IMAGE_TYPES: List[str] = Field(
        default=["image/jpeg", "image/png", "image/webp"]
    )
    ALLOWED_AUDIO_TYPES: List[str] = Field(
        default=["audio/mpeg", "audio/wav", "audio/x-wav", "audio/mp3"]
    )
    ALLOWED_VIDEO_TYPES: List[str] = Field(
        default=["video/mp4", "video/quicktime", "video/x-msvideo", "video/webm"]
    )
    
    # Result retention
    RESULT_RETENTION_PERIOD: int = Field(
        default=24 * 60 * 60  # 24 hours
    )
    
    # Processing settings
    DEFAULT_CONFIDENCE_THRESHOLD: float = Field(default=0.5)
    
    # Parse the BACKEND_CORS_ORIGINS from comma-separated string if needed
    @validator("BACKEND_CORS_ORIGINS", pre=True)
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)
    
    # Convert DEBUG string to boolean
    @validator("DEBUG", pre=True)
    def parse_debug(cls, v: Any) -> bool:
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.lower() == "true"
        return False
    
    # Convert CELERY_TASK_ALWAYS_EAGER string to boolean
    @validator("CELERY_TASK_ALWAYS_EAGER", pre=True)
    def parse_celery_eager(cls, v: Any) -> bool:
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.lower() == "true"
        return False
    
    class Config:
        case_sensitive = True
        env_file = ".env"
        env_file_encoding = "utf-8"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Build database URI
        if not self.SQLALCHEMY_DATABASE_URI:
            self.SQLALCHEMY_DATABASE_URI = (
                f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@"
                f"{self.POSTGRES_SERVER}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
            )
        
        # Ensure REDIS_URL is set correctly
        if not self.REDIS_URL:
            self.REDIS_URL = f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/0"
        
        # Ensure Celery URLs use REDIS_URL if not set
        if not self.CELERY_BROKER_URL:
            self.CELERY_BROKER_URL = self.REDIS_URL
        if not self.CELERY_RESULT_BACKEND:
            self.CELERY_RESULT_BACKEND = self.REDIS_URL


settings = Settings()