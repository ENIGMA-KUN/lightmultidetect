from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field, validator
from datetime import datetime
import uuid


class DetectionRequest(BaseModel):
    media_type: str = Field(..., description="Type of media: image, audio, video")
    detailed_analysis: bool = Field(False, description="Whether to perform detailed analysis")
    confidence_threshold: Optional[float] = Field(None, description="Confidence threshold for detection")
    
    @validator('media_type')
    def validate_media_type(cls, v):
        if v not in ['image', 'audio', 'video']:
            raise ValueError('media_type must be one of: image, audio, video')
        return v
    
    @validator('confidence_threshold')
    def validate_confidence_threshold(cls, v):
        if v is not None and (v < 0 or v > 1):
            raise ValueError('confidence_threshold must be between 0 and 1')
        return v


class DetectionResponse(BaseModel):
    task_id: str = Field(..., description="Task ID for tracking the detection process")
    status: str = Field(..., description="Status of the detection task")
    media_type: str = Field(..., description="Type of media being processed")
    estimated_time: int = Field(..., description="Estimated processing time in seconds")


class DetectionResult(BaseModel):
    id: uuid.UUID = Field(..., description="Unique identifier for the detection result")
    file_hash: str = Field(..., description="Hash of the analyzed file")
    media_type: str = Field(..., description="Type of media that was analyzed")
    is_fake: bool = Field(..., description="Whether the media is detected as fake")
    confidence_score: float = Field(..., description="Confidence score of the detection")
    created_at: datetime = Field(..., description="When the detection was performed")
    
    # Optional detailed fields
    detection_details: Optional[Dict[str, Any]] = Field(None, description="Detailed detection results")
    models_used: Optional[Dict[str, str]] = Field(None, description="Models used for detection")
    visualizations: Optional[Dict[str, str]] = Field(None, description="Paths to visualization files")
    
    class Config:
        orm_mode = True