from typing import Dict, Any, List, Optional, Union
from sqlalchemy.orm import Session
from uuid import UUID
import os
import json
from datetime import datetime, timedelta

from app.db.models import DetectionResult


class ResultService:
    """Service for handling detection results operations."""
    
    @staticmethod
    def get_result_by_id(result_id: Union[str, UUID], db: Session) -> Optional[DetectionResult]:
        """
        Get a result by its ID.
        
        Args:
            result_id: ID of the result
            db: Database session
            
        Returns:
            DetectionResult or None if not found
        """
        return db.query(DetectionResult).filter(DetectionResult.id == result_id).first()
    
    @staticmethod
    def get_recent_results(limit: int, db: Session) -> List[DetectionResult]:
        """
        Get recent detection results.
        
        Args:
            limit: Maximum number of results to return
            db: Database session
            
        Returns:
            List of DetectionResult objects
        """
        return db.query(DetectionResult).order_by(
            DetectionResult.created_at.desc()
        ).limit(limit).all()
    
    @staticmethod
    def get_results_by_media_type(
        media_type: str, 
        limit: int, 
        db: Session
    ) -> List[DetectionResult]:
        """
        Get results filtered by media type.
        
        Args:
            media_type: Type of media (image, audio, video)
            limit: Maximum number of results to return
            db: Database session
            
        Returns:
            List of DetectionResult objects
        """
        return db.query(DetectionResult).filter(
            DetectionResult.media_type == media_type
        ).order_by(
            DetectionResult.created_at.desc()
        ).limit(limit).all()
    
    @staticmethod
    def format_result_for_response(result: DetectionResult) -> Dict[str, Any]:
        """
        Format a database result object for API response.
        
        Args:
            result: DetectionResult object
            
        Returns:
            Dict containing formatted result data
        """
        visualizations = {}
        
        if result.heatmap_path:
            visualizations["heatmap"] = {
                "url": f"/visualizations/{os.path.basename(result.heatmap_path)}",
                "width": 512,
                "height": 512,
                "regions": result.detection_details.get("regions", [])
            }
        
        if result.temporal_analysis_path:
            temporal_data = result.detection_details.get("temporal_analysis", {})
            visualizations["temporal"] = {
                "timestamps": temporal_data.get("timestamps", []),
                "values": temporal_data.get("scores", []),
                "threshold": temporal_data.get("threshold", 0.5)
            }
        
        if "frequency_analysis" in result.detection_details:
            visualizations["frequency"] = result.detection_details["frequency_analysis"]
        
        return {
            "id": str(result.id),
            "is_fake": result.is_fake,
            "confidence_score": result.confidence_score,
            "media_type": result.media_type,
            "created_at": result.created_at.isoformat(),
            "detection_details": result.detection_details,
            "models_used": result.models_used,
            "visualizations": visualizations if visualizations else None
        }
    
    @staticmethod
    def generate_result_statistics(
        days: int, 
        db: Session
    ) -> Dict[str, Any]:
        """
        Generate statistics about detection results.
        
        Args:
            days: Number of days to include in statistics
            db: Database session
            
        Returns:
            Dict containing statistics
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        # Total results
        total_results = db.query(DetectionResult).filter(
            DetectionResult.created_at >= cutoff_date
        ).count()
        
        # Count by media type
        media_type_counts = {}
        for media_type in ['image', 'audio', 'video']:
            count = db.query(DetectionResult).filter(
                DetectionResult.created_at >= cutoff_date,
                DetectionResult.media_type == media_type
            ).count()
            media_type_counts[media_type] = count
        
        # Count real vs fake
        fake_count = db.query(DetectionResult).filter(
            DetectionResult.created_at >= cutoff_date,
            DetectionResult.is_fake == True
        ).count()
        
        real_count = db.query(DetectionResult).filter(
            DetectionResult.created_at >= cutoff_date,
            DetectionResult.is_fake == False
        ).count()
        
        # Average confidence score
        from sqlalchemy import func
        confidence_avg = db.query(
            func.avg(DetectionResult.confidence_score)
        ).filter(
            DetectionResult.created_at >= cutoff_date
        ).scalar() or 0
        
        return {
            "period_days": days,
            "total_results": total_results,
            "media_type_distribution": media_type_counts,
            "real_count": real_count,
            "fake_count": fake_count,
            "fake_percentage": (fake_count / total_results * 100) if total_results > 0 else 0,
            "average_confidence": float(confidence_avg)
        }