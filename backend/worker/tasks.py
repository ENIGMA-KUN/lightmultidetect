import os
import json
import logging
import uuid
from typing import Dict, List, Optional, Any
from pathlib import Path
import time

from celery import Celery
from celery.signals import task_prerun, task_postrun

from backend.core.config import settings
from backend.ml.inference import DeepfakeDetector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Celery app
celery_app = Celery(
    "deepfake_detection",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND
)

# Configure Celery
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 hour
)

# Global detector instance
_detector = None

def get_detector():
    """Get or initialize the deepfake detector."""
    global _detector
    if _detector is None:
        logger.info("Initializing DeepfakeDetector")
        model_path = os.path.join(settings.MODEL_WEIGHTS_DIR, "multimodal_detector.pt")
        _detector = DeepfakeDetector(
            model_path=model_path,
            confidence_threshold=0.5,
            temp_dir=os.path.join(settings.UPLOAD_DIR, "temp")
        )
    return _detector

@task_prerun.connect
def task_prerun_handler(task_id, task, *args, **kwargs):
    """Handler called before a task is run."""
    logger.info(f"Starting task {task.name}[{task_id}]")

@task_postrun.connect
def task_postrun_handler(task_id, task, retval, state, *args, **kwargs):
    """Handler called after a task is run."""
    logger.info(f"Task {task.name}[{task_id}] finished with state: {state}")

@celery_app.task(bind=True, name="process_detection")
def process_detection(
    self,
    task_id: str,
    file_paths: Dict[str, List[str]],
    confidence_threshold: float = 0.5,
    explain_results: bool = False
) -> Dict[str, Any]:
    """
    Process media files for deepfake detection.
    
    Args:
        task_id: Unique ID for the detection task
        file_paths: Dictionary with modality keys and file path lists
        confidence_threshold: Threshold for detection confidence
        explain_results: Whether to generate explanations
    
    Returns:
        Dictionary with detection results
    """
    task_start_time = time.time()
    
    # Create task directories
    task_dir = os.path.join(settings.RESULTS_DIR, task_id)
    vis_dir = os.path.join(settings.VISUALIZATIONS_DIR, task_id)
    os.makedirs(task_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    
    # Create status file
    status_file = os.path.join(task_dir, "status.json")
    with open(status_file, "w") as f:
        json.dump({
            "task_id": task_id,
            "status": "PROCESSING",
            "message": "Task is being processed",
            "start_time": task_start_time
        }, f)
    
    try:
        # Initialize the detector
        detector = get_detector()
        detector.confidence_threshold = confidence_threshold
        
        # Process files
        results = detector.process_multimodal(
            files=file_paths,
            explain_results=explain_results
        )
        
        # Add task ID
        results["task_id"] = task_id
        
        # Generate visualizations if required
        if explain_results:
            for result in results["results"]:
                if result["success"] and "explanation" in result:
                    vis_path = detector.generate_visualization(result, vis_dir)
                    if vis_path:
                        result["visualization_file"] = os.path.basename(vis_path)
        
        # Save results to file
        result_file = os.path.join(task_dir, "result.json")
        with open(result_file, "w") as f:
            json.dump(results, f, indent=2)
        
        # Update status file
        with open(status_file, "w") as f:
            json.dump({
                "task_id": task_id,
                "status": "COMPLETED",
                "message": "Task completed successfully",
                "start_time": task_start_time,
                "end_time": time.time(),
                "processing_time": time.time() - task_start_time
            }, f)
        
        return {
            "task_id": task_id,
            "status": "COMPLETED",
            "message": "Task completed successfully"
        }
        
    except Exception as e:
        logger.error(f"Error processing detection task {task_id}: {str(e)}")
        
        # Update status file
        with open(status_file, "w") as f:
            json.dump({
                "task_id": task_id,
                "status": "FAILED",
                "message": f"Error: {str(e)}",
                "start_time": task_start_time,
                "end_time": time.time()
            }, f)
        
        # Create error file
        error_file = os.path.join(task_dir, "error.txt")
        with open(error_file, "w") as f:
            f.write(f"Error processing task {task_id}: {str(e)}")
        
        return {
            "task_id": task_id,
            "status": "FAILED",
            "message": f"Error: {str(e)}"
        } 