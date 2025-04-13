import os
import logging
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from celery import Task

from app.db.session import SessionLocal
from app.db.models import DetectionResult
from app.core.config import settings
from app.tasks.celery_app import celery_app

# Configure logging
logger = logging.getLogger(__name__)


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
def cleanup_old_results(self):
    """
    Clean up detection results older than the retention period.
    Also removes associated files (uploads and visualizations).
    """
    try:
        logger.info("Starting cleanup of old detection results")
        
        # Calculate cutoff date based on retention period
        retention_period = settings.RESULT_RETENTION_PERIOD  # in seconds
        cutoff_date = datetime.utcnow() - timedelta(seconds=retention_period)
        
        # Find old results
        old_results = self.session.query(DetectionResult).filter(
            DetectionResult.created_at < cutoff_date
        ).all()
        
        if not old_results:
            logger.info("No old results to clean up")
            return {"status": "success", "cleaned": 0}
        
        cleaned_count = 0
        
        for result in old_results:
            try:
                # Remove associated files
                if result.file_path and os.path.exists(result.file_path):
                    os.remove(result.file_path)
                
                if result.heatmap_path and os.path.exists(result.heatmap_path):
                    os.remove(result.heatmap_path)
                
                if result.temporal_analysis_path and os.path.exists(result.temporal_analysis_path):
                    os.remove(result.temporal_analysis_path)
                
                # Delete the database record
                self.session.delete(result)
                cleaned_count += 1
                
            except Exception as e:
                logger.error(f"Error cleaning up result {result.id}: {str(e)}")
        
        # Commit changes
        self.session.commit()
        
        logger.info(f"Cleaned up {cleaned_count} old results")
        return {"status": "success", "cleaned": cleaned_count}
    
    except Exception as e:
        logger.error(f"Error in cleanup_old_results task: {str(e)}")
        return {"status": "error", "message": str(e)}


@celery_app.task(base=SQLAlchemyTask, bind=True)
def generate_result_statistics(self):
    """
    Generate statistics about detection results.
    This could be used for monitoring or dashboards.
    """
    try:
        # Get statistics for the last 24 hours
        one_day_ago = datetime.utcnow() - timedelta(days=1)
        
        # Total results in last 24 hours
        total_results = self.session.query(DetectionResult).filter(
            DetectionResult.created_at >= one_day_ago
        ).count()
        
        # Count by media type
        media_type_counts = {}
        for media_type in ['image', 'audio', 'video']:
            count = self.session.query(DetectionResult).filter(
                DetectionResult.created_at >= one_day_ago,
                DetectionResult.media_type == media_type
            ).count()
            media_type_counts[media_type] = count
        
        # Count real vs fake
        fake_count = self.session.query(DetectionResult).filter(
            DetectionResult.created_at >= one_day_ago,
            DetectionResult.is_fake == True
        ).count()
        
        real_count = self.session.query(DetectionResult).filter(
            DetectionResult.created_at >= one_day_ago,
            DetectionResult.is_fake == False
        ).count()
        
        # Average confidence score
        confidence_avg = self.session.query(
            func.avg(DetectionResult.confidence_score)
        ).filter(
            DetectionResult.created_at >= one_day_ago
        ).scalar() or 0
        
        return {
            "status": "success",
            "timestamp": datetime.utcnow().isoformat(),
            "period": "24h",
            "total_results": total_results,
            "media_type_counts": media_type_counts,
            "fake_count": fake_count,
            "real_count": real_count,
            "confidence_avg": float(confidence_avg)
        }
        
    except Exception as e:
        logger.error(f"Error in generate_result_statistics task: {str(e)}")
        return {"status": "error", "message": str(e)}