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