from typing import Dict, Any, Optional, List
from sqlalchemy.orm import Session

from app.db.models import DetectionResult
from app.models.ensemble import ensemble_detection


class DetectionService:
    """Service for handling deepfake detection operations."""

    @staticmethod
    def run_detection(
        file_path: str,
        file_hash: str,
        media_type: str,
        detailed_analysis: bool,
        confidence_threshold: float,
        db: Session
    ) -> Dict[str, Any]:
        """
        Run detection on a media file.
        
        Args:
            file_path: Path to the media file
            file_hash: Hash of the file for identification
            media_type: Type of media (image, audio, video)
            detailed_analysis: Whether to perform detailed analysis
            confidence_threshold: Threshold for detection confidence
            db: Database session
            
        Returns:
            Dict containing detection results
        """
        # Check if we already have results for this file hash
        existing_result = db.query(DetectionResult).filter(
            DetectionResult.file_hash == file_hash
        ).first()
        
        if existing_result:
            return {
                "id": str(existing_result.id),
                "is_fake": existing_result.is_fake,
                "confidence_score": existing_result.confidence_score,
                "detection_details": existing_result.detection_details,
                "models_used": existing_result.models_used,
                "media_type": existing_result.media_type,
                "heatmap_path": existing_result.heatmap_path,
                "temporal_analysis_path": existing_result.temporal_analysis_path
            }
        
        # Run ensemble detection
        result = ensemble_detection(
            file_path=file_path,
            media_type=media_type,
            detailed=detailed_analysis,
            confidence_threshold=confidence_threshold
        )
        
        return result
    
    @staticmethod
    def get_supported_media_types() -> Dict[str, List[str]]:
        """Get supported media types and their file extensions."""
        return {
            "image": [".jpg", ".jpeg", ".png"],
            "audio": [".mp3", ".wav", ".ogg"],
            "video": [".mp4", ".mov", ".avi"]
        }
    
    @staticmethod
    def get_model_information(media_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Get information about the detection models.
        
        Args:
            media_type: Optional filter for specific media type
            
        Returns:
            Dict containing model information
        """
        models = {
            "image": [
                {
                    "name": "XceptionNet",
                    "description": "State-of-the-art model for image deepfake detection",
                    "performance": {
                        "accuracy": 0.96,
                        "f1_score": 0.95
                    }
                },
                {
                    "name": "EfficientNet-B4",
                    "description": "Efficient model with excellent accuracy-to-computation ratio",
                    "performance": {
                        "accuracy": 0.94,
                        "f1_score": 0.93
                    }
                },
                {
                    "name": "MesoNet",
                    "description": "Lightweight model for fast inference",
                    "performance": {
                        "accuracy": 0.89,
                        "f1_score": 0.88
                    }
                }
            ],
            "audio": [
                {
                    "name": "Wav2Vec 2.0",
                    "description": "Advanced model for audio deepfake detection",
                    "performance": {
                        "accuracy": 0.92,
                        "f1_score": 0.91
                    }
                },
                {
                    "name": "RawNet2",
                    "description": "Raw waveform analysis for voice manipulation detection",
                    "performance": {
                        "accuracy": 0.90,
                        "f1_score": 0.89
                    }
                }
            ],
            "video": [
                {
                    "name": "3D-CNN",
                    "description": "Spatio-temporal analysis for video deepfakes",
                    "performance": {
                        "accuracy": 0.93,
                        "f1_score": 0.92
                    }
                },
                {
                    "name": "Two-Stream Network",
                    "description": "Dual-path network analyzing spatial and temporal features",
                    "performance": {
                        "accuracy": 0.91,
                        "f1_score": 0.90
                    }
                }
            ]
        }
        
        if media_type:
            return {media_type: models.get(media_type, [])}
        
        return models