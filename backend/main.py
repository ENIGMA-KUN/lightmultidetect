# backend/main.py
import os
import shutil
import time
import uuid
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException, Depends, Query, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import logging
from datetime import datetime
import asyncio

from backend.api.routes import router as api_router
from backend.core.config import settings
from backend.core.security import get_current_user

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="LightMultiDetect API",
    description="API for efficient deepfake detection across multiple modalities",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(api_router, prefix="/api")

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint returning API information
    """
    return {
        "message": "Welcome to LightMultiDetect API",
        "version": "1.0.0",
        "docs_url": "/docs",
    }


@app.on_event("startup")
async def startup_event():
    """
    Perform startup tasks
    """
    logger.info("Starting up LightMultiDetect API")
    
    # Create necessary directories
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    os.makedirs(settings.RESULTS_DIR, exist_ok=True)
    
    # Additional startup tasks can be added here


@app.on_event("shutdown")
async def shutdown_event():
    """
    Perform shutdown tasks
    """
    logger.info("Shutting down LightMultiDetect API")
    
    # Cleanup temporary files if needed
    if settings.CLEANUP_ON_SHUTDOWN:
        logger.info("Cleaning up temporary files")
        
        # Clean upload directory (only files older than 24 hours)
        cleanup_old_files(settings.UPLOAD_DIR, max_age_hours=24)
        
        # Clean results directory (only files older than 24 hours)
        cleanup_old_files(settings.RESULTS_DIR, max_age_hours=24)


def cleanup_old_files(directory: str, max_age_hours: int = 24):
    """
    Clean up old files in a directory
    
    Args:
        directory: Directory to clean
        max_age_hours: Maximum age of files in hours
    """
    if not os.path.exists(directory):
        return
        
    current_time = time.time()
    max_age_seconds = max_age_hours * 3600
    
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        
        # Skip directories
        if os.path.isdir(item_path):
            continue
            
        # Check file age
        file_age = current_time - os.path.getmtime(item_path)
        
        # Remove if older than max age
        if file_age > max_age_seconds:
            try:
                os.remove(item_path)
                logger.info(f"Removed old file: {item_path}")
            except Exception as e:
                logger.error(f"Error removing file {item_path}: {e}")


if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)


# backend/api/routes/__init__.py
from fastapi import APIRouter
from backend.api.routes import detection, users, health

router = APIRouter()

router.include_router(detection.router, prefix="/detection", tags=["Detection"])
router.include_router(users.router, prefix="/users", tags=["Users"])
router.include_router(health.router, prefix="/health", tags=["Health"])


# backend/api/routes/detection.py
import os
import shutil
import tempfile
import time
import uuid
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, File, UploadFile, BackgroundTasks, HTTPException, Depends, Query, Form, Path
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import logging
import asyncio
from celery.result import AsyncResult

from backend.core.security import get_current_user
from backend.core.config import settings
from backend.models.user import User
from backend.models.detection import DetectionRequest, DetectionResponse, DetectionStatus
from backend.tasks.detection_tasks import process_image, process_audio, process_video, process_multimodal

logger = logging.getLogger(__name__)

router = APIRouter()


class ModalityRequest(BaseModel):
    """Request for modality selection"""
    image: bool = True
    audio: bool = True
    video: bool = True


@router.post("/upload", response_model=DetectionResponse)
async def upload_media(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    modalities: Optional[str] = Form("all"),
    confidence_threshold: Optional[float] = Form(0.5),
    explain_results: Optional[bool] = Form(True),
    current_user: User = Depends(get_current_user)
):
    """
    Upload media for deepfake detection
    
    Args:
        background_tasks: Background tasks
        files: Media files to analyze
        modalities: Modalities to use ("all", "image", "audio", "video", or comma-separated)
        confidence_threshold: Confidence threshold for detection
        explain_results: Whether to include explanations in results
        current_user: Current authenticated user
    
    Returns:
        Detection response with task ID
    """
    # Validate modalities
    allowed_modalities = ["image", "audio", "video", "all"]
    
    if modalities == "all":
        selected_modalities = ["image", "audio", "video"]
    else:
        selected_modalities = [m.strip() for m in modalities.split(",")]
        invalid_modalities = [m for m in selected_modalities if m not in allowed_modalities]
        
        if invalid_modalities:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid modalities: {', '.join(invalid_modalities)}"
            )
    
    # Generate unique task ID
    task_id = str(uuid.uuid4())
    
    # Create directory for uploads
    task_dir = os.path.join(settings.UPLOAD_DIR, task_id)
    os.makedirs(task_dir, exist_ok=True)
    
    # Categorize and save uploaded files
    image_files = []
    audio_files = []
    video_files = []
    
    for file in files:
        # Determine file type
        file_ext = os.path.splitext(file.filename)[1].lower()
        file_path = os.path.join(task_dir, file.filename)
        
        # Save file
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        # Categorize file
        if file_ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            image_files.append(file_path)
        elif file_ext in [".mp3", ".wav", ".ogg", ".flac"]:
            audio_files.append(file_path)
        elif file_ext in [".mp4", ".avi", ".mov", ".mkv"]:
            video_files.append(file_path)
        else:
            # Skip unsupported files
            os.remove(file_path)
            logger.warning(f"Skipping unsupported file: {file.filename}")
    
    # Create detection request
    request = DetectionRequest(
        task_id=task_id,
        user_id=current_user.id,
        modalities=selected_modalities,
        confidence_threshold=confidence_threshold,
        explain_results=explain_results,
        image_files=image_files,
        audio_files=audio_files,
        video_files=video_files,
        created_at=time.time()
    )
    
    # Store request in database
    # This would be implemented to store the request details in a database
    
    # Process media based on selected modalities
    if "all" in selected_modalities:
        # Process multimodal detection
        task = process_multimodal.delay(
            task_id=task_id,
            image_files=image_files,
            audio_files=audio_files,
            video_files=video_files,
            confidence_threshold=confidence_threshold,
            explain_results=explain_results
        )
    else:
        # Process individual modalities
        if "image" in selected_modalities and image_files:
            task_image = process_image.delay(
                task_id=task_id,
                image_files=image_files,
                confidence_threshold=confidence_threshold,
                explain_results=explain_results
            )
        
        if "audio" in selected_modalities and audio_files:
            task_audio = process_audio.delay(
                task_id=task_id,
                audio_files=audio_files,
                confidence_threshold=confidence_threshold,
                explain_results=explain_results
            )
        
        if "video" in selected_modalities and video_files:
            task_video = process_video.delay(
                task_id=task_id,
                video_files=video_files,
                confidence_threshold=confidence_threshold,
                explain_results=explain_results
            )
    
    # Return response
    return DetectionResponse(
        task_id=task_id,
        status=DetectionStatus.PENDING,
        message="Media uploaded successfully, processing started"
    )


@router.get("/status/{task_id}", response_model=DetectionResponse)
async def check_status(
    task_id: str = Path(..., description="Task ID"),
    current_user: User = Depends(get_current_user)
):
    """
    Check status of deepfake detection task
    
    Args:
        task_id: Task ID
        current_user: Current authenticated user
    
    Returns:
        Detection response with status
    """
    # Check if task exists
    task_dir = os.path.join(settings.RESULTS_DIR, task_id)
    if not os.path.exists(task_dir):
        task_dir = os.path.join(settings.UPLOAD_DIR, task_id)
        if not os.path.exists(task_dir):
            raise HTTPException(
                status_code=404,
                detail=f"Task {task_id} not found"
            )
    
    # Check if result file exists
    result_file = os.path.join(settings.RESULTS_DIR, task_id, "result.json")
    if os.path.exists(result_file):
        # Task completed
        import json
        with open(result_file, "r") as f:
            result = json.load(f)
        
        return DetectionResponse(
            task_id=task_id,
            status=DetectionStatus.COMPLETED,
            message="Processing completed",
            results=result
        )
    
    # Check if error file exists
    error_file = os.path.join(settings.RESULTS_DIR, task_id, "error.txt")
    if os.path.exists(error_file):
        # Task failed
        with open(error_file, "r") as f:
            error_message = f.read()
        
        return DetectionResponse(
            task_id=task_id,
            status=DetectionStatus.FAILED,
            message=f"Processing failed: {error_message}"
        )
    
    # Task still processing
    return DetectionResponse(
        task_id=task_id,
        status=DetectionStatus.PROCESSING,
        message="Processing in progress"
    )


@router.get("/result/{task_id}", response_model=DetectionResponse)
async def get_result(
    task_id: str = Path(..., description="Task ID"),
    current_user: User = Depends(get_current_user)
):
    """
    Get result of deepfake detection task
    
    Args:
        task_id: Task ID
        current_user: Current authenticated user
    
    Returns:
        Detection response with results
    """
    # Check if task exists
    result_dir = os.path.join(settings.RESULTS_DIR, task_id)
    if not os.path.exists(result_dir):
        raise HTTPException(
            status_code=404,
            detail=f"Task {task_id} not found or still processing"
        )
    
    # Check if result file exists
    result_file = os.path.join(result_dir, "result.json")
    if not os.path.exists(result_file):
        raise HTTPException(
            status_code=404,
            detail=f"Results for task {task_id} not found"
        )
    
    # Return results
    import json
    with open(result_file, "r") as f:
        result = json.load(f)
    
    return DetectionResponse(
        task_id=task_id,
        status=DetectionStatus.COMPLETED,
        message="Processing completed",
        results=result
    )


@router.get("/visualization/{task_id}/{file_name}")
async def get_visualization(
    task_id: str = Path(..., description="Task ID"),
    file_name: str = Path(..., description="Visualization file name"),
    current_user: User = Depends(get_current_user)
):
    """
    Get visualization file for deepfake detection task
    
    Args:
        task_id: Task ID
        file_name: Visualization file name
        current_user: Current authenticated user
    
    Returns:
        Visualization file
    """
    # Check if task exists
    viz_dir = os.path.join(settings.RESULTS_DIR, task_id, "visualizations")
    if not os.path.exists(viz_dir):
        raise HTTPException(
            status_code=404,
            detail=f"Visualizations for task {task_id} not found"
        )
    
    # Check if file exists
    file_path = os.path.join(viz_dir, file_name)
    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=404,
            detail=f"Visualization file {file_name} not found"
        )
    
    # Return file
    return FileResponse(file_path)


# backend/api/routes/users.py
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
from typing import Optional

from backend.core.security import (
    authenticate_user,
    create_access_token,
    get_current_active_user,
    get_password_hash,
)
from backend.models.user import User, UserCreate, UserInDB, Token, UserUpdate

router = APIRouter()


@router.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Get access token for user authentication
    
    Args:
        form_data: OAuth2 password request form
    
    Returns:
        Access token
    """
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = create_access_token(data={"sub": user.username})
    
    return {"access_token": access_token, "token_type": "bearer"}


@router.post("/register", response_model=User)
async def register_user(user: UserCreate):
    """
    Register a new user
    
    Args:
        user: User creation data
    
    Returns:
        Created user
    """
    # Check if username already exists
    existing_user = get_user_by_username(user.username)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered",
        )
    
    # Hash password
    hashed_password = get_password_hash(user.password)
    
    # Create user
    db_user = UserInDB(
        username=user.username,
        email=user.email,
        hashed_password=hashed_password,
        disabled=False,
    )
    
    # Store user in database
    # This would be implemented to store the user in a database
    
    # Return user without hashed password
    return User(
        username=db_user.username,
        email=db_user.email,
        disabled=db_user.disabled,
    )


@router.get("/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    """
    Get current user information
    
    Args:
        current_user: Current authenticated user
    
    Returns:
        User information
    """
    return current_user


@router.put("/me", response_model=User)
async def update_user(
    user_update: UserUpdate,
    current_user: User = Depends(get_current_active_user)
):
    """
    Update current user information
    
    Args:
        user_update: User update data
        current_user: Current authenticated user
    
    Returns:
        Updated user information
    """
    # Update user in database
    # This would be implemented to update the user in a database
    
    # Return updated user
    updated_user = User(
        username=current_user.username,
        email=user_update.email or current_user.email,
        disabled=current_user.disabled,
    )
    
    return updated_user


# Helper functions (these would typically interact with a database)
def get_user_by_username(username: str) -> Optional[UserInDB]:
    """
    Get user by username
    
    Args:
        username: Username
    
    Returns:
        User if found, None otherwise
    """
    # This is a placeholder implementation
    # In a real application, this would query a database
    
    # For demo purposes, return None to indicate user doesn't exist
    return None


# backend/api/routes/health.py
from fastapi import APIRouter, Depends
from pydantic import BaseModel
import psutil
import time

from backend.core.security import get_current_admin_user
from backend.models.user import User

router = APIRouter()


class HealthStatus(BaseModel):
    """
    API health status
    """
    status: str
    version: str
    uptime: float
    cpu_usage: float
    memory_usage: float


@router.get("/", response_model=HealthStatus)
async def health_check():
    """
    Check API health status
    
    Returns:
        API health status
    """
    # Get system metrics
    cpu_usage = psutil.cpu_percent()
    memory = psutil.virtual_memory()
    memory_usage = memory.percent
    
    # Calculate uptime
    # In a real application, you would track the actual start time
    uptime = 0.0
    
    return HealthStatus(
        status="healthy",
        version="1.0.0",
        uptime=uptime,
        cpu_usage=cpu_usage,
        memory_usage=memory_usage
    )


class DetailedHealthStatus(HealthStatus):
    """
    Detailed API health status
    """
    disk_usage: float
    network_io: dict
    num_threads: int
    open_files: int


@router.get("/detailed", response_model=DetailedHealthStatus)
async def detailed_health_check(current_user: User = Depends(get_current_admin_user)):
    """
    Check detailed API health status (admin only)
    
    Args:
        current_user: Current authenticated admin user
    
    Returns:
        Detailed API health status
    """
    # Get system metrics
    cpu_usage = psutil.cpu_percent()
    memory = psutil.virtual_memory()
    memory_usage = memory.percent
    disk = psutil.disk_usage('/')
    disk_usage = disk.percent
    network = psutil.net_io_counters()
    network_io = {
        "bytes_sent": network.bytes_sent,
        "bytes_recv": network.bytes_recv,
        "packets_sent": network.packets_sent,
        "packets_recv": network.packets_recv,
    }
    num_threads = len(psutil.Process().threads())
    open_files = len(psutil.Process().open_files())
    
    # Calculate uptime
    # In a real application, you would track the actual start time
    uptime = 0.0
    
    return DetailedHealthStatus(
        status="healthy",
        version="1.0.0",
        uptime=uptime,
        cpu_usage=cpu_usage,
        memory_usage=memory_usage,
        disk_usage=disk_usage,
        network_io=network_io,
        num_threads=num_threads,
        open_files=open_files
    )


# backend/core/config.py
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


# backend/core/security.py
from datetime import datetime, timedelta
from typing import Optional, Union

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import ValidationError

from backend.core.config import settings
from backend.models.user import User, UserInDB, TokenData

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/users/token")

# User database (this would be replaced with a real database in production)
# This is a placeholder for demonstration purposes
fake_users_db = {
    "admin": {
        "username": "admin",
        "email": "admin@example.com",
        "hashed_password": pwd_context.hash("admin"),
        "disabled": False,
        "is_admin": True,
    },
    "user": {
        "username": "user",
        "email": "user@example.com",
        "hashed_password": pwd_context.hash("user"),
        "disabled": False,
        "is_admin": False,
    },
}


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify password against hash
    
    Args:
        plain_password: Plain password
        hashed_password: Hashed password
    
    Returns:
        True if password matches hash, False otherwise
    """
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """
    Get hash for password
    
    Args:
        password: Plain password
    
    Returns:
        Hashed password
    """
    return pwd_context.hash(password)


def get_user(username: str) -> Optional[UserInDB]:
    """
    Get user from database
    
    Args:
        username: Username
    
    Returns:
        User if found, None otherwise
    """
    if username in fake_users_db:
        user_dict = fake_users_db[username]
        return UserInDB(**user_dict)
    return None


def authenticate_user(username: str, password: str) -> Optional[UserInDB]:
    """
    Authenticate user
    
    Args:
        username: Username
        password: Password
    
    Returns:
        User if authentication successful, None otherwise
    """
    user = get_user(username)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user


def create_access_token(*, data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create access token
    
    Args:
        data: Token data
        expires_delta: Token expiration delta
    
    Returns:
        Encoded JWT token
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt


async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    """
    Get current user from token
    
    Args:
        token: JWT token
    
    Returns:
        Current user
    
    Raises:
        HTTPException: If token is invalid or user not found
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except (JWTError, ValidationError):
        raise credentials_exception
    
    user = get_user(username=token_data.username)
    if user is None:
        raise credentials_exception
    
    return User(
        username=user.username,
        email=user.email,
        disabled=user.disabled,
        is_admin=getattr(user, "is_admin", False)
    )


async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """
    Get current active user
    
    Args:
        current_user: Current user
    
    Returns:
        Current active user
    
    Raises:
        HTTPException: If user is disabled
    """
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


async def get_current_admin_user(current_user: User = Depends(get_current_active_user)) -> User:
    """
    Get current admin user
    
    Args:
        current_user: Current active user
    
    Returns:
        Current admin user
    
    Raises:
        HTTPException: If user is not an admin
    """
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions",
        )
    return current_user


# backend/models/user.py
from typing import Optional
from pydantic import BaseModel, EmailStr


class Token(BaseModel):
    """
    OAuth2 token
    """
    access_token: str
    token_type: str


class TokenData(BaseModel):
    """
    Token data
    """
    username: Optional[str] = None


class User(BaseModel):
    """
    User model
    """
    username: str
    email: Optional[str] = None
    disabled: Optional[bool] = None
    is_admin: Optional[bool] = False


class UserInDB(User):
    """
    User in database
    """
    hashed_password: str


class UserCreate(BaseModel):
    """
    User creation data
    """
    username: str
    email: EmailStr
    password: str


class UserUpdate(BaseModel):
    """
    User update data
    """
    email: Optional[EmailStr] = None
    password: Optional[str] = None


# backend/models/detection.py
from typing import List, Optional, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field


class DetectionStatus(str, Enum):
    """
    Detection status
    """
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class DetectionRequest(BaseModel):
    """
    Detection request
    """
    task_id: str
    user_id: str
    modalities: List[str]
    confidence_threshold: float = 0.5
    explain_results: bool = True
    image_files: List[str] = []
    audio_files: List[str] = []
    video_files: List[str] = []
    created_at: float


class DetectionResponse(BaseModel):
    """
    Detection response
    """
    task_id: str
    status: DetectionStatus
    message: str
    results: Optional[Dict[str, Any]] = None


class DetectionResult(BaseModel):
    """
    Detection result
    """
    file_name: str
    predicted_label: str
    confidence: float
    processing_time: float
    modality: str
    visualization_file: Optional[str] = None
    explanation: Optional[Dict[str, Any]] = None


# backend/tasks/detection_tasks.py
import os
import time
import json
import numpy as np
import torch
from typing import List, Dict, Any, Optional
import logging
from celery import Celery
from celery.utils.log import get_task_logger
import matplotlib.pyplot as plt
import cv2
from PIL import Image

from backend.core.config import settings

# Initialize Celery
celery_app = Celery(
    "detection_tasks",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND
)

# Configure Celery
celery_app.conf.task_routes = {
    "backend.tasks.detection_tasks.*": {"queue": "detection"}
}

# Configure logging
logger = get_task_logger(__name__)


@celery_app.task(bind=True, name="process_image")
def process_image(
    self,
    task_id: str,
    image_files: List[str],
    confidence_threshold: float = 0.5,
    explain_results: bool = True
) -> Dict[str, Any]:
    """
    Process images for deepfake detection
    
    Args:
        task_id: Task ID
        image_files: List of image file paths
        confidence_threshold: Confidence threshold for detection
        explain_results: Whether to include explanations in results
    
    Returns:
        Detection results
    """
    logger.info(f"Processing {len(image_files)} images for task {task_id}")
    
    try:
        # Create results directory
        results_dir = os.path.join(settings.RESULTS_DIR, task_id)
        os.makedirs(results_dir, exist_ok=True)
        
        # Create visualizations directory
        viz_dir = os.path.join(results_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Load model
        model_path = os.path.join(settings.MODEL_DIR, "image_model.pth")
        model = load_image_model(model_path)
        
        # Process each image
        results = []
        
        for i, image_file in enumerate(image_files):
            # Update progress
            self.update_state(
                state="PROGRESS",
                meta={"current": i, "total": len(image_files)}
            )
            
            # Process image
            start_time = time.time()
            result = process_single_image(
                image_file=image_file,
                model=model,
                confidence_threshold=confidence_threshold,
                explain=explain_results,
                viz_dir=viz_dir
            )
            end_time = time.time()
            
            # Add processing time
            result["processing_time"] = end_time - start_time
            
            # Add to results
            results.append(result)
        
        # Save results
        result_data = {
            "task_id": task_id,
            "modality": "image",
            "num_files": len(image_files),
            "results": results,
            "completed_at": time.time()
        }
        
        with open(os.path.join(results_dir, "result.json"), "w") as f:
            json.dump(result_data, f, indent=2)
        
        logger.info(f"Image processing completed for task {task_id}")
        
        return result_data
        
    except Exception as e:
        logger.error(f"Error processing images for task {task_id}: {e}")
        
        # Create error file
        with open(os.path.join(results_dir, "error.txt"), "w") as f:
            f.write(str(e))
        
        raise


@celery_app.task(bind=True, name="process_audio")
def process_audio(
    self,
    task_id: str,
    audio_files: List[str],
    confidence_threshold: float = 0.5,
    explain_results: bool = True
) -> Dict[str, Any]:
    """
    Process audio for deepfake detection
    
    Args:
        task_id: Task ID
        audio_files: List of audio file paths
        confidence_threshold: Confidence threshold for detection
        explain_results: Whether to include explanations in results
    
    Returns:
        Detection results
    """
    logger.info(f"Processing {len(audio_files)} audio files for task {task_id}")
    
    try:
        # Create results directory
        results_dir = os.path.join(settings.RESULTS_DIR, task_id)
        os.makedirs(results_dir, exist_ok=True)
        
        # Create visualizations directory
        viz_dir = os.path.join(results_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Load model
        model_path = os.path.join(settings.MODEL_DIR, "audio_model.pth")
        model = load_audio_model(model_path)
        
        # Process each audio file
        results = []
        
        for i, audio_file in enumerate(audio_files):
            # Update progress
            self.update_state(
                state="PROGRESS",
                meta={"current": i, "total": len(audio_files)}
            )
            
            # Process audio
            start_time = time.time()
            result = process_single_audio(
                audio_file=audio_file,
                model=model,
                confidence_threshold=confidence_threshold,
                explain=explain_results,
                viz_dir=viz_dir
            )
            end_time = time.time()
            
            # Add processing time
            result["processing_time"] = end_time - start_time
            
            # Add to results
            results.append(result)
        
        # Save results
        result_data = {
            "task_id": task_id,
            "modality": "audio",
            "num_files": len(audio_files),
            "results": results,
            "completed_at": time.time()
        }
        
        with open(os.path.join(results_dir, "result.json"), "w") as f:
            json.dump(result_data, f, indent=2)
        
        logger.info(f"Audio processing completed for task {task_id}")
        
        return result_data
        
    except Exception as e:
        logger.error(f"Error processing audio for task {task_id}: {e}")
        
        # Create error file
        with open(os.path.join(results_dir, "error.txt"), "w") as f:
            f.write(str(e))
        
        raise


@celery_app.task(bind=True, name="process_video")
def process_video(
    self,
    task_id: str,
    video_files: List[str],
    confidence_threshold: float = 0.5,
    explain_results: bool = True
) -> Dict[str, Any]:
    """
    Process videos for deepfake detection
    
    Args:
        task_id: Task ID
        video_files: List of video file paths
        confidence_threshold: Confidence threshold for detection
        explain_results: Whether to include explanations in results
    
    Returns:
        Detection results
    """
    logger.info(f"Processing {len(video_files)} videos for task {task_id}")
    
    try:
        # Create results directory
        results_dir = os.path.join(settings.RESULTS_DIR, task_id)
        os.makedirs(results_dir, exist_ok=True)
        
        # Create visualizations directory
        viz_dir = os.path.join(results_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Load model
        model_path = os.path.join(settings.MODEL_DIR, "video_model.pth")
        model = load_video_model(model_path)
        
        # Process each video
        results = []
        
        for i, video_file in enumerate(video_files):
            # Update progress
            self.update_state(
                state="PROGRESS",
                meta={"current": i, "total": len(video_files)}
            )
            
            # Process video
            start_time = time.time()
            result = process_single_video(
                video_file=video_file,
                model=model,
                confidence_threshold=confidence_threshold,
                explain=explain_results,
                viz_dir=viz_dir
            )
            end_time = time.time()
            
            # Add processing time
            result["processing_time"] = end_time - start_time
            
            # Add to results
            results.append(result)
        
        # Save results
        result_data = {
            "task_id": task_id,
            "modality": "video",
            "num_files": len(video_files),
            "results": results,
            "completed_at": time.time()
        }
        
        with open(os.path.join(results_dir, "result.json"), "w") as f:
            json.dump(result_data, f, indent=2)
        
        logger.info(f"Video processing completed for task {task_id}")
        
        return result_data
        
    except Exception as e:
        logger.error(f"Error processing videos for task {task_id}: {e}")
        
        # Create error file
        with open(os.path.join(results_dir, "error.txt"), "w") as f:
            f.write(str(e))
        
        raise


@celery_app.task(bind=True, name="process_multimodal")
def process_multimodal(
    self,
    task_id: str,
    image_files: List[str],
    audio_files: List[str],
    video_files: List[str],
    confidence_threshold: float = 0.5,
    explain_results: bool = True
) -> Dict[str, Any]:
    """
    Process media with multi-modal deepfake detection
    
    Args:
        task_id: Task ID
        image_files: List of image file paths
        audio_files: List of audio file paths
        video_files: List of video file paths
        confidence_threshold: Confidence threshold for detection
        explain_results: Whether to include explanations in results
    
    Returns:
        Detection results
    """
    logger.info(f"Processing multi-modal data for task {task_id}")
    
    try:
        # Create results directory
        results_dir = os.path.join(settings.RESULTS_DIR, task_id)
        os.makedirs(results_dir, exist_ok=True)
        
        # Create visualizations directory
        viz_dir = os.path.join(results_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Load multi-modal model
        model_path = os.path.join(settings.MODEL_DIR, "multimodal_model.pth")
        model = load_multimodal_model(model_path)
        
        # Process each media type
        all_results = []
        
        # Process images
        if image_files:
            for i, image_file in enumerate(image_files):
                # Update progress
                self.update_state(
                    state="PROGRESS",
                    meta={
                        "current": i,
                        "total": len(image_files) + len(audio_files) + len(video_files),
                        "type": "image"
                    }
                )
                
                # Process image
                start_time = time.time()
                result = process_single_image(
                    image_file=image_file,
                    model=model,
                    confidence_threshold=confidence_threshold,
                    explain=explain_results,
                    viz_dir=viz_dir
                )
                end_time = time.time()
                
                # Add processing time
                result["processing_time"] = end_time - start_time
                
                # Add to results
                all_results.append(result)
        
        # Process audio files
        if audio_files:
            for i, audio_file in enumerate(audio_files):
                # Update progress
                self.update_state(
                    state="PROGRESS",
                    meta={
                        "current": len(image_files) + i,
                        "total": len(image_files) + len(audio_files) + len(video_files),
                        "type": "audio"
                    }
                )
                
                # Process audio
                start_time = time.time()
                result = process_single_audio(
                    audio_file=audio_file,
                    model=model,
                    confidence_threshold=confidence_threshold,
                    explain=explain_results,
                    viz_dir=viz_dir
                )
                end_time = time.time()
                
                # Add processing time
                result["processing_time"] = end_time - start_time
                
                # Add to results
                all_results.append(result)
        
        # Process video files
        if video_files:
            for i, video_file in enumerate(video_files):
                # Update progress
                self.update_state(
                    state="PROGRESS",
                    meta={
                        "current": len(image_files) + len(audio_files) + i,
                        "total": len(image_files) + len(audio_files) + len(video_files),
                        "type": "video"
                    }
                )
                
                # Process video
                start_time = time.time()
                result = process_single_video(
                    video_file=video_file,
                    model=model,
                    confidence_threshold=confidence_threshold,
                    explain=explain_results,
                    viz_dir=viz_dir
                )
                end_time = time.time()
                
                # Add processing time
                result["processing_time"] = end_time - start_time
                
                # Add to results
                all_results.append(result)
        
        # Save results
        result_data = {
            "task_id": task_id,
            "modality": "multimodal",
            "num_files": {
                "image": len(image_files),
                "audio": len(audio_files),
                "video": len(video_files),
                "total": len(image_files) + len(audio_files) + len(video_files)
            },
            "results": all_results,
            "completed_at": time.time()
        }
        
        with open(os.path.join(results_dir, "result.json"), "w") as f:
            json.dump(result_data, f, indent=2)
        
        logger.info(f"Multi-modal processing completed for task {task_id}")
        
        return result_data
        
    except Exception as e:
        logger.error(f"Error processing multi-modal data for task {task_id}: {e}")
        
        # Create error file
        with open(os.path.join(results_dir, "error.txt"), "w") as f:
            f.write(str(e))
        
        raise


# Helper functions (to be implemented with actual model loading and inference)
def load_image_model(model_path: str):
    """Load image model (placeholder)"""
    # This would be implemented to load the actual model
    return "image_model"


def load_audio_model(model_path: str):
    """Load audio model (placeholder)"""
    # This would be implemented to load the actual model
    return "audio_model"


def load_video_model(model_path: str):
    """Load video model (placeholder)"""
    # This would be implemented to load the actual model
    return "video_model"


def load_multimodal_model(model_path: str):
    """Load multi-modal model (placeholder)"""
    # This would be implemented to load the actual model
    return "multimodal_model"


def process_single_image(
    image_file: str,
    model,
    confidence_threshold: float,
    explain: bool,
    viz_dir: str
) -> Dict[str, Any]:
    """
    Process a single image for deepfake detection (placeholder)
    
    In a real implementation, this would:
    1. Load the image
    2. Preprocess it
    3. Run inference with the model
    4. Generate explanations if requested
    5. Create visualizations
    6. Return results
    """
    # Simulate processing delay
    time.sleep(0.5)
    
    # Create a random prediction for demonstration
    is_fake = np.random.random() > 0.5
    confidence = np.random.uniform(0.7, 0.95)
    
    # File name without path
    file_name = os.path.basename(image_file)
    
    # Create visualization
    viz_file = None
    if explain:
        viz_file = f"{os.path.splitext(file_name)[0]}_viz.png"
        viz_path = os.path.join(viz_dir, viz_file)
        
        # Create a simple visualization (placeholder)
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        img = plt.imread(image_file)
        plt.imshow(img)
        plt.title("Original Image")
        
        plt.subplot(1, 2, 2)
        # Generate a heatmap (placeholder)
        heatmap = np.random.rand(img.shape[0], img.shape[1])
        plt.imshow(img)
        plt.imshow(heatmap, cmap='jet', alpha=0.5)
        plt.title("Deepfake Detection Heatmap")
        
        plt.suptitle(f"Prediction: {'Fake' if is_fake else 'Real'} ({confidence:.2f})")
        plt.tight_layout()
        plt.savefig(viz_path)
        plt.close()
    
    # Create explanation
    explanation = None
    if explain:
        explanation = {
            "important_regions": [
                {
                    "x": int(np.random.uniform(0, 1) * 100),
                    "y": int(np.random.uniform(0, 1) * 100),
                    "width": int(np.random.uniform(20, 50)),
                    "height": int(np.random.uniform(20, 50)),
                    "importance": np.random.uniform(0.7, 0.9)
                }
            ],
            "frequency_analysis": {
                "low_freq_energy": np.random.uniform(0.2, 0.4),
                "mid_freq_energy": np.random.uniform(0.3, 0.5),
                "high_freq_energy": np.random.uniform(0.1, 0.3)
            }
        }
    
    return {
        "file_name": file_name,
        "predicted_label": "fake" if is_fake else "real",
        "confidence": float(confidence),
        "modality": "image",
        "visualization_file": viz_file,
        "explanation": explanation
    }


def process_single_audio(
    audio_file: str,
    model,
    confidence_threshold: float,
    explain: bool,
    viz_dir: str
) -> Dict[str, Any]:
    """
    Process a single audio file for deepfake detection (placeholder)
    
    In a real implementation, this would:
    1. Load the audio
    2. Preprocess it
    3. Run inference with the model
    4. Generate explanations if requested
    5. Create visualizations
    6. Return results
    """
    # Simulate processing delay
    time.sleep(0.7)
    
    # Create a random prediction for demonstration
    is_fake = np.random.random() > 0.5
    confidence = np.random.uniform(0.7, 0.95)
    
    # File name without path
    file_name = os.path.basename(audio_file)
    
    # Create visualization
    viz_file = None
    if explain:
        viz_file = f"{os.path.splitext(file_name)[0]}_viz.png"
        viz_path = os.path.join(viz_dir, viz_file)
        
        # Create a simple visualization (placeholder)
        plt.figure(figsize=(10, 6))
        
        # Generate a spectrogram (placeholder)
        t = np.linspace(0, 10, 1000)
        freq = np.linspace(0, 1000, 200)
        spectrogram = np.random.rand(len(freq), len(t))
        
        plt.imshow(spectrogram, aspect='auto', origin='lower', extent=[0, 10, 0, 1000])
        plt.colorbar(label='Magnitude')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.title(f"Audio Spectrogram - Prediction: {'Fake' if is_fake else 'Real'} ({confidence:.2f})")
        plt.tight_layout()
        plt.savefig(viz_path)
        plt.close()
    
    # Create explanation
    explanation = None
    if explain:
        explanation = {
            "temporal_consistency": np.random.uniform(0.5, 0.9),
            "spectral_artifacts": {
                "presence": np.random.choice([True, False]),
                "strength": np.random.uniform(0.1, 0.8)
            },
            "frequency_bands": {
                "low": np.random.uniform(0.2, 0.4),
                "mid": np.random.uniform(0.3, 0.5),
                "high": np.random.uniform(0.1, 0.3)
            }
        }
    
    return {
        "file_name": file_name,
        "predicted_label": "fake" if is_fake else "real",
        "confidence": float(confidence),
        "modality": "audio",
        "visualization_file": viz_file,
        "explanation": explanation
    }


def process_single_video(
    video_file: str,
    model,
    confidence_threshold: float,
    explain: bool,
    viz_dir: str
) -> Dict[str, Any]:
    """
    Process a single video for deepfake detection (placeholder)
    
    In a real implementation, this would:
    1. Load the video
    2. Extract frames and audio
    3. Preprocess them
    4. Run inference with the model
    5. Generate explanations if requested
    6. Create visualizations
    7. Return results
    """
    # Simulate processing delay
    time.sleep(1.5)
    
    # Create a random prediction for demonstration
    is_fake = np.random.random() > 0.5
    confidence = np.random.uniform(0.7, 0.95)
    
    # File name without path
    file_name = os.path.basename(video_file)
    
    # Create visualization
    viz_file = None
    if explain:
        viz_file = f"{os.path.splitext(file_name)[0]}_viz.png"
        viz_path = os.path.join(viz_dir, viz_file)
        
        # Create a simple visualization (placeholder)
        plt.figure(figsize=(15, 8))
        
        # Simulate video frames
        n_frames = 6
        for i in range(n_frames):
            plt.subplot(2, 3, i+1)
            
            # Generate a random frame (placeholder)
            frame = np.random.rand(100, 100, 3)
            plt.imshow(frame)
            plt.title(f"Frame {i+1}")
            plt.axis('off')
        
        plt.suptitle(f"Video Analysis - Prediction: {'Fake' if is_fake else 'Real'} ({confidence:.2f})")
        plt.tight_layout()
        plt.savefig(viz_path)
        plt.close()
    
    # Create explanation
    explanation = None
    if explain:
        explanation = {
            "temporal_consistency": np.random.uniform(0.5, 0.9),
            "facial_landmarks": {
                "stability": np.random.uniform(0.6, 0.95),
                "abnormalities": np.random.choice([True, False])
            },
            "audio_visual_sync": np.random.uniform(0.7, 0.95),
            "important_frames": [
                {
                    "frame_idx": int(np.random.uniform(0, 100)),
                    "confidence": np.random.uniform(0.7, 0.9)
                },
                {
                    "frame_idx": int(np.random.uniform(100, 200)),
                    "confidence": np.random.uniform(0.7, 0.9)
                }
            ]
        }
    
    return {
        "file_name": file_name,
        "predicted_label": "fake" if is_fake else "real",
        "confidence": float(confidence),
        "modality": "video",
        "visualization_file": viz_file,
        "explanation": explanation
    }