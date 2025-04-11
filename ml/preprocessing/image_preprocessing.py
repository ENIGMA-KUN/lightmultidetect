# ml/preprocessing/image_preprocessing.py
import os
import cv2
import numpy as np
from tqdm import tqdm
import mediapipe as mp


class ImagePreprocessor:
    """
    Preprocessor for image-based deepfake detection
    """
    def __init__(self, face_size=224, min_face_confidence=0.5):
        """
        Initialize preprocessor
        
        Args:
            face_size: Target size for face images
            min_face_confidence: Minimum confidence for face detection
        """
        self.face_size = face_size
        self.min_face_confidence = min_face_confidence
        
        # Initialize MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,  # 0 for close-range, 1 for far-range
            min_detection_confidence=min_face_confidence
        )
    
    def detect_faces(self, image):
        """
        Detect faces in an image
        
        Args:
            image: RGB image array
            
        Returns:
            List of face detections with bounding boxes and confidence scores
        """
        # Convert image to RGB if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        # Detect faces
        results = self.face_detection.process(image)
        
        if not results.detections:
            return []
        
        # Extract face detections
        faces = []
        height, width = image.shape[:2]
        
        for detection in results.detections:
            # Get bounding box
            bbox = detection.location_data.relative_bounding_box
            x = int(bbox.xmin * width)
            y = int(bbox.ymin * height)
            w = int(bbox.width * width)
            h = int(bbox.height * height)
            
            # Ensure coordinates are within image bounds
            x = max(0, x)
            y = max(0, y)
            w = min(width - x, w)
            h = min(height - y, h)
            
            faces.append({
                'bbox': (x, y, w, h),
                'confidence': detection.score[0]
            })
        
        return faces
    
    def extract_faces(self, image, confidence_threshold=0.9):
        """
        Extract faces from an image
        
        Args:
            image: RGB image array
            confidence_threshold: Confidence threshold for face detection
            
        Returns:
            List of extracted face images
        """
        # Detect faces
        faces = self.detect_faces(image)
        
        # Extract face regions
        face_images = []
        for face in faces:
            if face['confidence'] >= confidence_threshold:
                x, y, w, h = face['bbox']
                face_img = image[y:y+h, x:x+w]
                
                # Resize face image
                face_img = cv2.resize(face_img, (self.face_size, self.face_size))
                face_images.append(face_img)
        
        return face_images
    
    def preprocess_image(self, image):
        """
        Preprocess an image for deepfake detection
        
        Args:
            image: RGB image array
            
        Returns:
            Preprocessed image array
        """
        # Convert to RGB if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        # Resize image
        image = cv2.resize(image, (self.face_size, self.face_size))
        
        # Normalize pixel values
        image = image.astype(np.float32) / 255.0
        
        return image
    
    def process_image(self, image_path, output_dir=None, save_faces=False):
        """
        Process an image file for deepfake detection
        
        Args:
            image_path: Path to image file
            output_dir: Directory to save processed data
            save_faces: Whether to save extracted faces
            
        Returns:
            Dictionary with processed image data
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create output directory if needed
        if save_faces and output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
        
        # Extract faces
        faces = self.extract_faces(image)
        
        # Save faces if requested
        if save_faces and output_dir is not None:
            for face_idx, face in enumerate(faces):
                face_path = os.path.join(
                    output_dir,
                    f"face_{face_idx:06d}.jpg"
                )
                cv2.imwrite(face_path, cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
        
        # Preprocess full image
        processed_image = self.preprocess_image(image)
        
        return {
            'image': processed_image,
            'faces': faces
        }
    
    def process_directory(self, input_dir, output_dir, file_extensions=('.jpg', '.jpeg', '.png')):
        """
        Process all image files in a directory
        
        Args:
            input_dir: Input directory
            output_dir: Output directory
            file_extensions: File extensions to process
            
        Returns:
            Dictionary with processing statistics
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all image files
        image_files = []
        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.lower().endswith(file_extensions):
                    image_files.append(os.path.join(root, file))
        
        # Process image files
        stats = {
            'total_files': len(image_files),
            'processed_files': 0,
            'failed_files': 0,
            'total_faces': 0
        }
        
        for image_path in tqdm(image_files, desc="Processing image files"):
            try:
                # Create image-specific output directory
                image_output_dir = os.path.join(
                    output_dir,
                    os.path.splitext(os.path.basename(image_path))[0]
                )
                
                # Process image
                result = self.process_image(
                    image_path=image_path,
                    output_dir=image_output_dir,
                    save_faces=True
                )
                
                # Update stats
                stats['processed_files'] += 1
                stats['total_faces'] += len(result['faces'])
                
            except Exception as e:
                print(f"Error processing image {image_path}: {e}")
                stats['failed_files'] += 1
        
        return stats