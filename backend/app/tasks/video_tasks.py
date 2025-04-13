import os
import uuid
from celery import Task
from sqlalchemy.orm import Session

from app.db.session import SessionLocal
from app.db.models import DetectionResult
from app.models.ensemble import ensemble_detection
from app.utils.visualization import generate_temporal_visualization
from app.utils.video_utils import extract_frames
from app.tasks.celery_app import celery_app


class SQLAlchemyTask(Task):
    """Base Task with SQLAlchemy session handling."""
    _session = None
    
    def after_return(self, *args, **kwargs):
        if self._session is not None:
            self._session.close()
    
    @property
    def session(self) -> Session:
        if self._session is None:
            self._session = SessionLocal()
        return self._session


@celery_app.task(base=SQLAlchemyTask, bind=True)
def detect_video(self, video_path: str, file_hash: str, detailed: bool = False, confidence_threshold: float = 0.5):
    """
    Process a video for deepfake detection.
    
    Args:
        self: Task instance (injected by Celery)
        video_path (str): Path to the video file
        file_hash (str): Hash of the file for identification
        detailed (bool): Whether to perform detailed analysis
        confidence_threshold (float): Threshold for detection confidence
    
    Returns:
        str: ID of the result record
    """
    try:
        # Update task status
        self.update_state(state="PROGRESS", meta={"progress": 10, "message": "Starting video analysis"})
        
        # Extract frames for initial analysis
        self.update_state(state="PROGRESS", meta={"progress": 20, "message": "Extracting video frames"})
        
        # Perform ensemble detection
        self.update_state(state="PROGRESS", meta={"progress": 30, "message": "Running detection models"})
        result = ensemble_detection(video_path, "video", detailed, confidence_threshold)
        
        self.update_state(state="PROGRESS", meta={"progress": 60, "message": "Analyzing results"})
        
        # Generate visualizations if detailed analysis
        temporal_analysis_path = None
        
        if detailed and not result.get("error") and "temporal_analysis" in result.get("detection_details", {}):
            self.update_state(state="PROGRESS", meta={"progress": 70, "message": "Generating visualizations"})
            
            try:
                # Get temporal analysis data
                temporal_data = result["detection_details"]["temporal_analysis"]
                
                # Generate visualization
                filename = f"temporal_{uuid.uuid4().hex}.png"
                output_path = os.path.join("visualizations", filename)
                
                temporal_analysis_path = generate_temporal_visualization(
                    temporal_data["timestamps"],
                    temporal_data["scores"],
                    temporal_data["threshold"],
                    "Video Deepfake Confidence Over Time",
                    output_path
                )
            except Exception as e:
                self.update_state(state="PROGRESS", meta={"progress": 75, "message": f"Visualization error: {str(e)}"})
        
        # Create database record
        self.update_state(state="PROGRESS", meta={"progress": 80, "message": "Saving results"})
        
        db_result = DetectionResult(
            id=uuid.uuid4(),
            file_hash=file_hash,
            file_path=video_path,
            media_type="video",
            is_fake=result["is_fake"],
            confidence_score=result["confidence_score"],
            detection_details=result.get("detection_details", {}),
            models_used=result.get("models_used", {}),
            temporal_analysis_path=temporal_analysis_path
        )
        
        self.session.add(db_result)
        self.session.commit()
        
        self.update_state(state="PROGRESS", meta={"progress": 100, "message": "Analysis complete"})
        
        return str(db_result.id)
    
    except Exception as e:
        # Log the error
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Error in detect_video task: {str(e)}")
        
        # Create error record in database
        try:
            db_result = DetectionResult(
                id=uuid.uuid4(),
                file_hash=file_hash,
                file_path=video_path,
                media_type="video",
                is_fake=False,
                confidence_score=0.0,
                detection_details={"error": str(e)},
                models_used={}
            )
            
            self.session.add(db_result)
            self.session.commit()
            
            return str(db_result.id)
        except Exception as db_error:
            logger.error(f"Error saving failure record: {str(db_error)}")
            raise e