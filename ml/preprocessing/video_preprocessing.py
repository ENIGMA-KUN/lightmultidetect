# ml/preprocessing/video_preprocessing.py
import os
import cv2
import numpy as np
from tqdm import tqdm
import mediapipe as mp
from typing import List, Tuple, Dict, Any


class VideoPreprocessor:
    """
    Preprocessor for video-based deepfake detection
    """
    def __init__(
        self,
        frame_interval: int = 30,
        face_size: Tuple[int, int] = (224, 224),
        min_face_confidence: float = 0.5
    ):
        """
        Initialize preprocessor
        
        Args:
            frame_interval: Number of frames to skip between extractions
            face_size: Target size for face images (height, width)
            min_face_confidence: Minimum confidence for face detection
        """
        self.frame_interval = frame_interval
        self.face_size = face_size
        self.min_face_confidence = min_face_confidence
        
        # Initialize MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,  # 0 for close-range, 1 for far-range
            min_detection_confidence=min_face_confidence
        )
    
    def extract_frames(
        self,
        video_path: str,
        output_dir: str = None,
        save_frames: bool = False
    ) -> List[np.ndarray]:
        """
        Extract frames from video file
        
        Args:
            video_path: Path to video file
            output_dir: Directory to save extracted frames
            save_frames: Whether to save extracted frames
            
        Returns:
            List of extracted frames
        """
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Calculate frame indices to extract
        frame_indices = np.arange(0, total_frames, self.frame_interval)
        
        # Extract frames
        frames = []
        for frame_idx in tqdm(frame_indices, desc="Extracting frames"):
            # Set frame position
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            
            # Read frame
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Save frame if requested
            if save_frames and output_dir is not None:
                os.makedirs(output_dir, exist_ok=True)
                frame_path = os.path.join(
                    output_dir,
                    f"frame_{frame_idx:06d}.jpg"
                )
                cv2.imwrite(frame_path, frame)
            
            frames.append(frame_rgb)
        
        # Release video capture
        cap.release()
        
        return frames
    
    def detect_faces(
        self,
        frame: np.ndarray
    ) -> List[Dict[str, Any]]:
        """
        Detect faces in frame
        
        Args:
            frame: RGB frame
            
        Returns:
            List of detected faces with bounding boxes and confidence scores
        """
        # Detect faces
        results = self.face_detection.process(frame)
        
        faces = []
        if results.detections:
            height, width = frame.shape[:2]
            
            for detection in results.detections:
                # Get bounding box
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * width)
                y = int(bbox.ymin * height)
                w = int(bbox.width * width)
                h = int(bbox.height * height)
                
                # Get confidence score
                confidence = detection.score[0]
                
                faces.append({
                    'bbox': (x, y, w, h),
                    'confidence': confidence
                })
        
        return faces
    
    def extract_faces(
        self,
        frame: np.ndarray,
        faces: List[Dict[str, Any]]
    ) -> List[np.ndarray]:
        """
        Extract face regions from frame
        
        Args:
            frame: RGB frame
            faces: List of detected faces
            
        Returns:
            List of extracted face images
        """
        face_images = []
        
        for face in faces:
            # Get bounding box
            x, y, w, h = face['bbox']
            
            # Extract face region
            face_img = frame[y:y+h, x:x+w]
            
            # Resize face image
            face_img = cv2.resize(face_img, self.face_size)
            
            face_images.append(face_img)
        
        return face_images
    
    def process_video(
        self,
        video_path: str,
        output_dir: str = None,
        save_frames: bool = False,
        save_faces: bool = False
    ) -> Dict[str, Any]:
        """
        Process video file for deepfake detection
        
        Args:
            video_path: Path to video file
            output_dir: Directory to save processed data
            save_frames: Whether to save extracted frames
            save_faces: Whether to save extracted faces
            
        Returns:
            Dictionary with processed video data
        """
        # Create output directories if needed
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            if save_frames:
                os.makedirs(os.path.join(output_dir, 'frames'), exist_ok=True)
            if save_faces:
                os.makedirs(os.path.join(output_dir, 'faces'), exist_ok=True)
        
        # Extract frames
        frames = self.extract_frames(
            video_path=video_path,
            output_dir=os.path.join(output_dir, 'frames') if save_frames else None,
            save_frames=save_frames
        )
        
        # Process frames
        processed_data = {
            'frames': [],
            'faces': [],
            'metadata': {
                'total_frames': len(frames),
                'faces_detected': 0
            }
        }
        
        for frame_idx, frame in enumerate(frames):
            # Detect faces
            faces = self.detect_faces(frame)
            
            # Extract faces
            face_images = self.extract_faces(frame, faces)
            
            # Update metadata
            processed_data['metadata']['faces_detected'] += len(faces)
            
            # Save faces if requested
            if save_faces and output_dir is not None:
                for face_idx, face_img in enumerate(face_images):
                    face_path = os.path.join(
                        output_dir,
                        'faces',
                        f"frame_{frame_idx:06d}_face_{face_idx:03d}.jpg"
                    )
                    cv2.imwrite(face_path, cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))
            
            # Store processed data
            processed_data['frames'].append({
                'frame_idx': frame_idx,
                'faces': faces,
                'face_images': face_images
            })
        
        return processed_data
    
    def process_directory(
        self,
        input_dir: str,
        output_dir: str,
        file_extensions: Tuple[str, ...] = ('.mp4', '.avi', '.mov')
    ) -> Dict[str, Any]:
        """
        Process all video files in a directory
        
        Args:
            input_dir: Input directory
            output_dir: Output directory
            file_extensions: File extensions to process
            
        Returns:
            Dictionary with processing statistics
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all video files
        video_files = []
        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.lower().endswith(file_extensions):
                    video_files.append(os.path.join(root, file))
        
        # Process video files
        stats = {
            'total_files': len(video_files),
            'processed_files': 0,
            'failed_files': 0,
            'total_frames': 0,
            'total_faces': 0
        }
        
        for video_path in tqdm(video_files, desc="Processing video files"):
            try:
                # Create video-specific output directory
                video_output_dir = os.path.join(
                    output_dir,
                    os.path.splitext(os.path.basename(video_path))[0]
                )
                
                # Process video
                processed_data = self.process_video(
                    video_path=video_path,
                    output_dir=video_output_dir,
                    save_frames=True,
                    save_faces=True
                )
                
                # Update stats
                stats['processed_files'] += 1
                stats['total_frames'] += processed_data['metadata']['total_frames']
                stats['total_faces'] += processed_data['metadata']['faces_detected']
                
            except Exception as e:
                print(f"Error processing video {video_path}: {e}")
                stats['failed_files'] += 1
        
        return stats 