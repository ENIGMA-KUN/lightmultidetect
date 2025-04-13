import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple

from app.models.image_models import detect_deepfake
from app.models.audio_models import detect_deepfake_audio
from app.models.video_models import detect_deepfake_video

# Configure logging
logger = logging.getLogger(__name__)


def ensemble_detection(
    file_path: str, 
    media_type: str, 
    detailed: bool = False, 
    confidence_threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Perform ensemble deepfake detection using multiple models.
    
    Args:
        file_path (str): Path to the media file
        media_type (str): Type of media (image, audio, video)
        detailed (bool): Whether to perform detailed analysis
        confidence_threshold (float): Threshold for detection confidence
    
    Returns:
        Dict[str, Any]: Detection results
    """
    try:
        result = {
            "is_fake": False,
            "confidence_score": 0.0,
            "media_type": media_type,
            "models_used": {},
            "detection_details": {}
        }
        
        if media_type == "image":
            # Use multiple image models for ensemble
            models = ["xception", "efficientnet", "mesonet"]
            model_weights = [0.5, 0.3, 0.2]  # Weights for each model
            
            model_results = []
            
            for model_type in models:
                detection_result = detect_deepfake(file_path, model_type, detailed)
                model_results.append(detection_result)
                
                # Store model info
                result["models_used"][model_type] = detection_result.get("model_used", model_type)
            
            # Calculate weighted confidence score
            confidence_scores = [r["confidence_score"] for r in model_results]
            weighted_score = sum(s * w for s, w in zip(confidence_scores, model_weights))
            
            # Determine if fake based on threshold
            result["is_fake"] = weighted_score >= confidence_threshold
            result["confidence_score"] = weighted_score
            
            # Add detailed results if requested
            if detailed:
                result["detection_details"] = {
                    "model_scores": {model: score for model, score in zip(models, confidence_scores)},
                    "frequency_analysis": model_results[0].get("detailed_analysis", {}).get("artifact_detection", {}),
                    "regions": []  # Would be populated with suspicious regions
                }
                
                # Add more detailed analysis if available
                if "detailed_analysis" in model_results[0]:
                    result["detection_details"].update(model_results[0]["detailed_analysis"])
        
        elif media_type == "audio":
            # Use multiple audio models for ensemble
            models = ["wav2vec2", "rawnet2", "melspec"]
            model_weights = [0.5, 0.3, 0.2]  # Weights for each model
            
            model_results = []
            
            for model_type in models:
                try:
                    detection_result = detect_deepfake_audio(file_path, model_type, detailed)
                    model_results.append(detection_result)
                    
                    # Store model info
                    result["models_used"][model_type] = detection_result.get("model_used", model_type)
                except Exception as e:
                    logger.error(f"Error with audio model {model_type}: {str(e)}")
                    # Skip this model
            
            # Only proceed if we have results
            if model_results:
                # Calculate weighted confidence score
                confidence_scores = [r["confidence_score"] for r in model_results]
                adjusted_weights = model_weights[:len(model_results)]
                adjusted_weights = [w / sum(adjusted_weights) for w in adjusted_weights]  # Normalize weights
                weighted_score = sum(s * w for s, w in zip(confidence_scores, adjusted_weights))
                
                # Determine if fake based on threshold
                result["is_fake"] = weighted_score >= confidence_threshold
                result["confidence_score"] = weighted_score
                
                # Add detailed results if requested
                if detailed:
                    result["detection_details"] = {
                        "model_scores": {model: score for model, score in zip(models[:len(model_results)], confidence_scores)},
                    }
                    
                    # Add more detailed analysis if available
                    if "detailed_analysis" in model_results[0]:
                        result["detection_details"].update(model_results[0]["detailed_analysis"])
            else:
                result["error"] = "All audio models failed to process the file"
        
        elif media_type == "video":
            # For video, we'll use a combined approach:
            # 1. Use video models for temporal analysis
            # 2. Use image models on key frames
            # 3. Use audio models on the extracted audio if available
            
            # 1. Video model analysis
            video_models = ["3dcnn", "timesformer"]
            video_weights = [0.6, 0.4]
            video_results = []
            
            for model_type in video_models:
                try:
                    detection_result = detect_deepfake_video(file_path, model_type, detailed)
                    video_results.append(detection_result)
                    
                    # Store model info
                    result["models_used"][f"video_{model_type}"] = detection_result.get("model_used", model_type)
                except Exception as e:
                    logger.error(f"Error with video model {model_type}: {str(e)}")
                    # Skip this model
            
            # 2. Image model on key frames (indirectly through video model)
            # This is handled within the video model
            
            # 3. Audio model on extracted audio (optional)
            # This would require audio extraction from video
            # For simplicity, we'll skip this in the example
            
            # Calculate combined score from video models
            if video_results:
                confidence_scores = [r["confidence_score"] for r in video_results]
                adjusted_weights = video_weights[:len(video_results)]
                adjusted_weights = [w / sum(adjusted_weights) for w in adjusted_weights]  # Normalize weights
                weighted_score = sum(s * w for s, w in zip(confidence_scores, adjusted_weights))
                
                # Determine if fake based on threshold
                result["is_fake"] = weighted_score >= confidence_threshold
                result["confidence_score"] = weighted_score
                
                # Add detailed results if requested
                if detailed:
                    result["detection_details"] = {
                        "model_scores": {model: score for model, score in zip(video_models[:len(video_results)], confidence_scores)},
                    }
                    
                    # Add more detailed analysis if available
                    if "detailed_analysis" in video_results[0]:
                        result["detection_details"].update(video_results[0]["detailed_analysis"])
            else:
                result["error"] = "All video models failed to process the file"
        
        else:
            result["error"] = f"Unsupported media type: {media_type}"
        
        return result
        
    except Exception as e:
        logger.error(f"Error during ensemble detection: {str(e)}")
        return {
            "error": str(e),
            "is_fake": False,
            "confidence_score": 0.0,
            "media_type": media_type
        }