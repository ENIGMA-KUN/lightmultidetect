import os
import uuid
from celery import Task
from sqlalchemy.orm import Session

from app.db.session import SessionLocal
from app.db.models import DetectionResult
from app.models.ensemble import ensemble_detection
from app.utils.visualization import generate_heatmap_visualization
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
def detect_image(self, image_path: str, file_hash: str, detailed: bool = False, confidence_threshold: float = 0.5):
    """
    Process an image for deepfake detection.
    
    Args:
        self: Task instance (injected by Celery)
        image_path (str): Path to the image file
        file_hash (str): Hash of the file for identification
        detailed (bool): Whether to perform detailed analysis
        confidence_threshold (float): Threshold for detection confidence
    
    Returns:
        str: ID of the result record
    """
    try:
        # Update task status
        self.update_state(state="PROGRESS", meta={"progress": 10, "message": "Starting image analysis"})
        
        # Perform ensemble detection
        self.update_state(state="PROGRESS", meta={"progress": 20, "message": "Running detection models"})
        result = ensemble_detection(image_path, "image", detailed, confidence_threshold)
        
        self.update_state(state="PROGRESS", meta={"progress": 60, "message": "Analyzing results"})
        
        # Generate visualizations if detailed analysis
        heatmap_path = None
        
        if detailed and not result.get("error"):
            from app.models.image_models import generate_heatmap
            
            self.update_state(state="PROGRESS", meta={"progress": 70, "message": "Generating visualizations"})
            
            try:
                # Generate heatmap
                image_np, heatmap_np = generate_heatmap(image_path)
                
                # Save visualization
                filename = f"heatmap_{uuid.uuid4().hex}.png"
                output_path = os.path.join("visualizations", filename)
                
                heatmap_path = generate_heatmap_visualization(image_np, heatmap_np, output_path)
            except Exception as e:
                self.update_state(state="PROGRESS", meta={"progress": 75, "message": f"Visualization error: {str(e)}"})
        
        # Create database record
        self.update_state(state="PROGRESS", meta={"progress": 80, "message": "Saving results"})
        
        db_result = DetectionResult(
            id=uuid.uuid4(),
            file_hash=file_hash,
            file_path=image_path,
            media_type="image",
            is_fake=result["is_fake"],
            confidence_score=result["confidence_score"],
            detection_details=result.get("detection_details", {}),
            models_used=result.get("models_used", {}),
            heatmap_path=heatmap_path
        )
        
        self.session.add(db_result)
        self.session.commit()
        
        self.update_state(state="PROGRESS", meta={"progress": 100, "message": "Analysis complete"})
        
        return str(db_result.id)
    
    except Exception as e:
        # Log the error
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Error in detect_image task: {str(e)}")
        
        # Create error record in database
        try:
            db_result = DetectionResult(
                id=uuid.uuid4(),
                file_hash=file_hash,
                file_path=image_path,
                media_type="image",
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