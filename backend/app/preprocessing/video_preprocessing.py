import os
import cv2
import numpy as np
import torch
from typing import List, Tuple, Dict, Any, Optional, Union
import logging
from pathlib import Path
import tempfile

# Import from other preprocessing modules
from app.preprocessing.image_preprocessing import detect_faces, preprocess_for_model as preprocess_image
from app.preprocessing.audio_preprocessing import extract_audio_from_video, analyze_voice_consistency

# Configure logging
logger = logging.getLogger(__name__)


def get_video_info(video_path: str) -> Dict[str, Any]:
    """
    Get basic information about a video file.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Dictionary with video information
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        
        # Read first frame to check codec
        ret, first_frame = cap.read()
        has_frame = ret
        
        # Release the video
        cap.release()
        
        return {
            "fps": float(fps),
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "duration": float(duration),
            "has_frame": has_frame,
            "aspect_ratio": width / height if height > 0 else 0
        }
    except Exception as e:
        logger.error(f"Error getting video info for {video_path}: {str(e)}")
        return {
            "error": str(e),
            "fps": 0,
            "frame_count": 0,
            "width": 0,
            "height": 0,
            "duration": 0,
            "has_frame": False
        }


def extract_frames(video_path: str, max_frames: int = 30, interval: Optional[int] = None) -> List[np.ndarray]:
    """
    Extract frames from a video file at regular intervals.
    
    Args:
        video_path: Path to the video file
        max_frames: Maximum number of frames to extract
        interval: Interval between frames (if None, calculated automatically)
        
    Returns:
        List of extracted frames as numpy arrays (BGR format)
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return []
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate interval to evenly distribute frames
        if interval is None:
            interval = max(1, total_frames // max_frames)
        
        frames = []
        frame_indices = []
        
        # Extract frames at intervals
        for i in range(0, total_frames, interval):
            if len(frames) >= max_frames:
                break
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            
            if not ret:
                break
            
            frames.append(frame)
            frame_indices.append(i)
        
        cap.release()
        
        logger.info(f"Extracted {len(frames)} frames from video {video_path}")
        return frames
    
    except Exception as e:
        logger.error(f"Error extracting frames from {video_path}: {str(e)}")
        return []


def extract_faces_from_video(video_path: str, max_frames: int = 30, save_faces: bool = False, 
                           output_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Extract faces from video frames.
    
    Args:
        video_path: Path to the video file
        max_frames: Maximum number of frames to process
        save_faces: Whether to save extracted faces to disk
        output_dir: Directory to save faces (if save_faces is True)
        
    Returns:
        Dictionary with face detection results
    """
    try:
        # Extract frames
        frames = extract_frames(video_path, max_frames)
        if not frames:
            return {"error": "No frames extracted", "face_count": 0}
        
        # Process each frame
        face_results = []
        total_faces = 0
        
        for i, frame in enumerate(frames):
            # Detect faces in frame
            faces = detect_faces(frame)
            
            # Count faces
            face_count = len(faces)
            total_faces += face_count
            
            # Create result for this frame
            frame_result = {
                "frame_index": i,
                "face_count": face_count,
                "faces": []
            }
            
            # Process each face
            for j, face in enumerate(faces):
                face_data = {
                    "height": face.shape[0],
                    "width": face.shape[1]
                }
                
                # Save face if requested
                if save_faces:
                    if output_dir is None:
                        output_dir = os.path.join(os.path.dirname(video_path), "faces")
                    
                    os.makedirs(output_dir, exist_ok=True)
                    face_filename = f"frame_{i:04d}_face_{j:02d}.jpg"
                    face_path = os.path.join(output_dir, face_filename)
                    cv2.imwrite(face_path, face)
                    face_data["path"] = face_path
                
                frame_result["faces"].append(face_data)
            
            face_results.append(frame_result)
        
        return {
            "total_frames_processed": len(frames),
            "total_faces_detected": total_faces,
            "average_faces_per_frame": total_faces / len(frames) if frames else 0,
            "frame_results": face_results
        }
    
    except Exception as e:
        logger.error(f"Error extracting faces from video {video_path}: {str(e)}")
        return {"error": str(e), "face_count": 0}


def calculate_optical_flow(prev_frame: np.ndarray, curr_frame: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Calculate optical flow between two consecutive frames.
    
    Args:
        prev_frame: Previous frame as numpy array
        curr_frame: Current frame as numpy array
        
    Returns:
        Tuple of (flow_magnitude, mean_magnitude)
    """
    try:
        # Convert frames to grayscale
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None, 
            pyr_scale=0.5, levels=3, winsize=15, 
            iterations=3, poly_n=5, poly_sigma=1.2, 
            flags=0
        )
        
        # Calculate magnitude of flow vectors
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mean_magnitude = np.mean(magnitude)
        
        return magnitude, float(mean_magnitude)
    
    except Exception as e:
        logger.error(f"Error calculating optical flow: {str(e)}")
        return np.zeros_like(prev_frame[:, :, 0]), 0.0


def analyze_temporal_consistency(video_path: str, max_frames: int = 30) -> Dict[str, Any]:
    """
    Analyze temporal consistency between video frames.
    
    Args:
        video_path: Path to the video file
        max_frames: Maximum number of frames to analyze
        
    Returns:
        Dictionary with temporal consistency analysis
    """
    try:
        # Extract frames
        frames = extract_frames(video_path, max_frames)
        
        if len(frames) < 2:
            return {
                "consistency_score": 1.0,
                "error": "Not enough frames for temporal analysis"
            }
        
        # Calculate optical flow between consecutive frames
        flow_magnitudes = []
        
        for i in range(len(frames) - 1):
            _, mean_magnitude = calculate_optical_flow(frames[i], frames[i+1])
            flow_magnitudes.append(mean_magnitude)
        
        # Calculate statistics
        mean_flow = np.mean(flow_magnitudes)
        std_flow = np.std(flow_magnitudes)
        
        # Calculate consistency score
        # Higher variability in flow might indicate inconsistent manipulation
        consistency_score = 1.0 / (1.0 + std_flow)
        
        return {
            "consistency_score": float(consistency_score),
            "mean_flow": float(mean_flow),
            "std_flow": float(std_flow),
            "flow_magnitudes": [float(m) for m in flow_magnitudes],
            "frames_analyzed": len(frames)
        }
    
    except Exception as e:
        logger.error(f"Error analyzing temporal consistency for {video_path}: {str(e)}")
        return {
            "error": str(e),
            "consistency_score": 0.0
        }


def analyze_video_noise(video_path: str, max_frames: int = 30) -> Dict[str, Any]:
    """
    Analyze video noise patterns for deepfake detection.
    
    Args:
        video_path: Path to the video file
        max_frames: Maximum number of frames to analyze
        
    Returns:
        Dictionary with noise analysis results
    """
    try:
        # Extract frames
        frames = extract_frames(video_path, max_frames)
        
        if not frames:
            return {"error": "No frames extracted"}
        
        # Analyze noise in each frame
        noise_levels = []
        
        for frame in frames:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply Laplacian for edge detection
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            
            # Calculate statistics of Laplacian
            std_dev = np.std(laplacian)
            noise_levels.append(std_dev)
        
        # Calculate statistics across frames
        mean_noise = np.mean(noise_levels)
        std_noise = np.std(noise_levels)
        
        # Calculate noise consistency (low variation in noise levels)
        noise_consistency = 1.0 / (1.0 + std_noise)
        
        return {
            "mean_noise_level": float(mean_noise),
            "noise_level_std": float(std_noise),
            "noise_consistency": float(noise_consistency),
            "noise_levels": [float(n) for n in noise_levels],
            "frames_analyzed": len(frames)
        }
    
    except Exception as e:
        logger.error(f"Error analyzing video noise for {video_path}: {str(e)}")
        return {"error": str(e)}


def analyze_lip_sync(video_path: str, max_frames: int = 30) -> Dict[str, Any]:
    """
    Analyze lip synchronization with audio for deepfake detection.
    
    Args:
        video_path: Path to the video file
        max_frames: Maximum number of frames to analyze
        
    Returns:
        Dictionary with lip sync analysis results
    """
    try:
        # This is a complex task that would ideally require specialized models
        # This is a simplified placeholder implementation
        
        # Extract frames
        frames = extract_frames(video_path, max_frames)
        
        if not frames:
            return {
                "lip_sync_score": 0.5,
                "confidence": 0.0,
                "error": "No frames extracted"
            }
        
        # Extract audio
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
            temp_audio_path = temp_audio.name
        
        try:
            audio_path = extract_audio_from_video(video_path, temp_audio_path)
            audio_analysis = analyze_voice_consistency(audio_path)
        except Exception as e:
            logger.warning(f"Error extracting or analyzing audio: {str(e)}")
            audio_analysis = {"consistency_score": 0.5}
            
        # Detect faces in all frames
        face_frames = []
        for frame in frames:
            faces = detect_faces(frame)
            if faces:
                face_frames.append(faces[0])  # Use the first face
        
        if not face_frames:
            return {
                "lip_sync_score": 0.5,
                "confidence": 0.0,
                "error": "No faces detected in frames"
            }
        
        # In a real implementation, we would:
        # 1. Detect lip landmarks in each frame
        # 2. Track lip movement patterns
        # 3. Correlate with audio features
        # 4. Analyze synchronization
        
        # For now, return placeholder values
        # The audio consistency can be a weak proxy for lip sync
        audio_consistency = audio_analysis.get("consistency_score", 0.5)
        
        return {
            "lip_sync_score": max(0.1, min(0.9, audio_consistency)),  # Scale between 0.1-0.9
            "confidence": 0.65,  # Low confidence since this is simplified
            "frames_analyzed": len(face_frames),
            "audio_consistency": audio_consistency
        }
    
    except Exception as e:
        logger.error(f"Error analyzing lip sync for {video_path}: {str(e)}")
        return {
            "lip_sync_score": 0.5,
            "confidence": 0.0,
            "error": str(e)
        }
    finally:
        # Clean up temporary audio file
        if 'temp_audio_path' in locals() and os.path.exists(temp_audio_path):
            try:
                os.unlink(temp_audio_path)
            except:
                pass


def preprocess_for_model(video_path: str, model_type: str = '3dcnn', max_frames: int = 32) -> Dict[str, torch.Tensor]:
    """
    Preprocess video for specific deepfake detection models.
    
    Args:
        video_path: Path to the video file
        model_type: Type of model ('3dcnn', 'two-stream', 'timesformer')
        max_frames: Maximum number of frames to process
        
    Returns:
        Dictionary of preprocessed tensors for the model
    """
    try:
        # Extract frames from video
        frames = extract_frames(video_path, max_frames=max_frames)
        
        if not frames:
            raise ValueError(f"No frames could be extracted from {video_path}")
        
        # Convert to numpy arrays and normalize
        frame_tensor = np.array(frames) / 255.0
        
        if model_type.lower() == '3dcnn':
            # 3DCNN expects input shape [batch, channels, frames, height, width]
            frame_tensor = np.transpose(frame_tensor, (3, 0, 1, 2))  # [C, T, H, W]
            frame_tensor = np.expand_dims(frame_tensor, axis=0)  # [1, C, T, H, W]
            return {"video": torch.tensor(frame_tensor, dtype=torch.float32)}
        
        elif model_type.lower() == 'two-stream':
            # Two-stream expects spatial and temporal inputs
            # Spatial: a single frame
            spatial_tensor = frame_tensor[0]  # First frame
            spatial_tensor = np.transpose(spatial_tensor, (2, 0, 1))  # [C, H, W]
            spatial_tensor = np.expand_dims(spatial_tensor, axis=0)  # [1, C, H, W]
            
            # Temporal: optical flow between frames
            flow_tensor = np.zeros((len(frame_tensor) - 1, frame_tensor.shape[1], frame_tensor.shape[2], 2), dtype=np.float32)
            
            # Calculate optical flow
            for i in range(len(frame_tensor) - 1):
                prev_frame = (frame_tensor[i] * 255).astype(np.uint8)
                curr_frame = (frame_tensor[i+1] * 255).astype(np.uint8)
                
                # Convert to grayscale
                prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
                curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_RGB2GRAY)
                
                # Calculate optical flow
                flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                
                # Normalize flow
                flow = flow / (np.max(np.abs(flow)) + 1e-8)
                
                # Store flow
                flow_tensor[i] = flow
            
            # Take mean of all flows for simplicity
            mean_flow = np.mean(flow_tensor, axis=0)
            mean_flow = np.transpose(mean_flow, (2, 0, 1))  # [2, H, W]
            mean_flow = np.expand_dims(mean_flow, axis=0)  # [1, 2, H, W]
            
            return {
                "spatial": torch.tensor(spatial_tensor, dtype=torch.float32),
                "temporal": torch.tensor(mean_flow, dtype=torch.float32)
            }
        
        elif model_type.lower() == 'timesformer':
            # TimeSformer expects input shape [batch, channels, frames, height, width]
            frame_tensor = np.transpose(frame_tensor, (3, 0, 1, 2))  # [C, T, H, W]
            frame_tensor = np.expand_dims(frame_tensor, axis=0)  # [1, C, T, H, W]
            return {"video": torch.tensor(frame_tensor, dtype=torch.float32)}
        
        else:
            raise ValueError(f"Unsupported video model type: {model_type}")
    
    except Exception as e:
        logger.error(f"Error preprocessing video for model {model_type}: {str(e)}")
        raise ValueError(f"Failed to preprocess video: {str(e)}")


def comprehensive_video_analysis(video_path: str, analyze_audio: bool = True) -> Dict[str, Any]:
    """
    Perform comprehensive analysis of a video for deepfake detection.
    
    Args:
        video_path: Path to the video file
        analyze_audio: Whether to analyze audio as well
        
    Returns:
        Dictionary with comprehensive analysis results
    """
    results = {}
    
    try:
        # Get basic video info
        results["video_info"] = get_video_info(video_path)
        
        # Analyze temporal consistency
        results["temporal_analysis"] = analyze_temporal_consistency(video_path)
        
        # Analyze noise patterns
        results["noise_analysis"] = analyze_video_noise(video_path)
        
        # Analyze face detections
        face_results = extract_faces_from_video(video_path, max_frames=20)
        results["face_analysis"] = face_results
        
        # Analyze lip sync
        results["lip_sync_analysis"] = analyze_lip_sync(video_path)
        
        # Analyze audio if requested
        if analyze_audio:
            try:
                # Extract audio
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
                    temp_audio_path = temp_audio.name
                
                audio_path = extract_audio_from_video(video_path, temp_audio_path)
                
                # Analyze voice consistency
                from app.preprocessing.audio_preprocessing import analyze_voice_consistency, extract_audio_features
                
                results["voice_consistency"] = analyze_voice_consistency(audio_path)
                results["audio_features"] = extract_audio_features(audio_path)
                
                # Clean up
                os.unlink(temp_audio_path)
            except Exception as e:
                logger.error(f"Error in audio analysis: {str(e)}")
                results["audio_error"] = str(e)
        
        return results
    
    except Exception as e:
        logger.error(f"Error in comprehensive video analysis: {str(e)}")
        results["error"] = str(e)
        return results