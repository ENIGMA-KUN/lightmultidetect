import os
import cv2
import numpy as np
from typing import List, Tuple, Optional
import logging

# Configure logging
logger = logging.getLogger(__name__)


def is_valid_video(video_path: str) -> bool:
    """
    Check if the file is a valid video file.
    
    Args:
        video_path (str): Path to the video file
    
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False
        
        # Read the first frame
        ret, frame = cap.read()
        cap.release()
        
        return ret
    except Exception as e:
        logger.error(f"Invalid video file: {str(e)}")
        return False


def extract_frames(video_path: str, max_frames: int = 30, interval: Optional[int] = None) -> List[np.ndarray]:
    """
    Extract frames from a video file.
    
    Args:
        video_path (str): Path to the video file
        max_frames (int): Maximum number of frames to extract
        interval (int, optional): Interval between frames to extract
    
    Returns:
        List[np.ndarray]: List of extracted frames
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return []
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Determine frame interval
        if interval is None:
            # Calculate interval to evenly distribute frames
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
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            frame_indices.append(i)
        
        cap.release()
        
        logger.info(f"Extracted {len(frames)} frames from video")
        return frames
    
    except Exception as e:
        logger.error(f"Error extracting frames: {str(e)}")
        return []


def extract_faces(frames: List[np.ndarray]) -> List[np.ndarray]:
    """
    Extract faces from a list of frames.
    
    Args:
        frames (List[np.ndarray]): List of frames
    
    Returns:
        List[np.ndarray]: List of extracted face frames
    """
    try:
        # Load face detector
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        face_frames = []
        
        for frame in frames:
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            
            # Detect faces
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            # If no faces found, use the whole frame
            if len(faces) == 0:
                face_frames.append(frame)
                continue
            
            # Get the largest face
            largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
            x, y, w, h = largest_face
            
            # Add some margin
            margin = int(0.2 * w)
            x = max(0, x - margin)
            y = max(0, y - margin)
            w = min(frame.shape[1] - x, w + 2 * margin)
            h = min(frame.shape[0] - y, h + 2 * margin)
            
            # Extract face with margin
            face_frame = frame[y:y+h, x:x+w]
            face_frames.append(face_frame)
        
        return face_frames
    
    except Exception as e:
        logger.error(f"Error extracting faces: {str(e)}")
        return frames  # Return original frames if face extraction fails


def analyze_temporal_consistency(frames: List[np.ndarray]) -> dict:
    """
    Analyze temporal consistency between frames.
    
    Args:
        frames (List[np.ndarray]): List of frames
    
    Returns:
        dict: Temporal consistency analysis
    """
    try:
        if len(frames) < 2:
            return {
                "consistency_score": 1.0,
                "error": "Not enough frames for temporal analysis"
            }
        
        # Calculate optical flow between consecutive frames
        flow_magnitudes = []
        
        for i in range(len(frames) - 1):
            prev_frame = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
            curr_frame = cv2.cvtColor(frames[i+1], cv2.COLOR_RGB2GRAY)
            
            # Calculate optical flow
            flow = cv2.calcOpticalFlowFarneback(prev_frame, curr_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            
            # Calculate magnitude of flow vectors
            magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            
            # Get average magnitude
            mean_magnitude = np.mean(magnitude)
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
            "flow_magnitudes": [float(m) for m in flow_magnitudes]
        }
    
    except Exception as e:
        logger.error(f"Error analyzing temporal consistency: {str(e)}")
        return {
            "consistency_score": 0.0,
            "error": str(e)
        }


def analyze_lip_sync(video_path: str) -> dict:
    """
    Analyze lip synchronization with audio.
    
    Note: This is a placeholder function. Real lip sync analysis would require
    more advanced techniques involving audio-visual synchronization.
    
    Args:
        video_path (str): Path to the video file
    
    Returns:
        dict: Lip sync analysis
    """
    try:
        # Placeholder implementation
        # A real implementation would:
        # 1. Extract audio track
        # 2. Extract video frames focusing on mouth region
        # 3. Analyze correlation between audio features and mouth movements
        
        # Return dummy values for now
        return {
            "lip_sync_score": 0.75,  # Higher is better
            "confidence": 0.65
        }
    
    except Exception as e:
        logger.error(f"Error analyzing lip sync: {str(e)}")
        return {
            "lip_sync_score": 0.5,
            "confidence": 0.0,
            "error": str(e)
        }