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
        user_id=current_user.username,
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