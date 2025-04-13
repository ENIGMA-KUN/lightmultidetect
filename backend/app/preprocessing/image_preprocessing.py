import os
import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from typing import List, Tuple, Dict, Any, Optional, Union
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Define standard transforms for different models
XCEPTION_TRANSFORM = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

EFFICIENTNET_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

MESONET_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


def load_image(image_path: str) -> Image.Image:
    """
    Load an image from file path.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        PIL Image object
    """
    try:
        img = Image.open(image_path).convert('RGB')
        return img
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {str(e)}")
        raise ValueError(f"Failed to load image: {str(e)}")


def detect_faces(image_path: str, min_face_size: int = 20) -> List[np.ndarray]:
    """
    Detect faces in an image using OpenCV's Haar Cascade.
    
    Args:
        image_path: Path to the image file
        min_face_size: Minimum face size to detect
        
    Returns:
        List of face regions as numpy arrays
    """
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        logger.error(f"Could not read image: {image_path}")
        return []
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Load face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(min_face_size, min_face_size)
    )
    
    # Extract face regions
    face_regions = []
    for (x, y, w, h) in faces:
        face_regions.append(img[y:y+h, x:x+w])
    
    logger.info(f"Detected {len(face_regions)} faces in {image_path}")
    return face_regions


def preprocess_for_model(image: Union[str, Image.Image, np.ndarray], model_type: str = 'xception') -> torch.Tensor:
    """
    Preprocess an image for the specified model.
    
    Args:
        image: Image as file path, PIL Image, or numpy array
        model_type: Type of model ('xception', 'efficientnet', 'mesonet')
        
    Returns:
        Preprocessed image as a torch.Tensor with batch dimension
    """
    # Load image if path is provided
    if isinstance(image, str):
        image = load_image(image)
    
    # Convert numpy array to PIL Image
    if isinstance(image, np.ndarray):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Select appropriate transform
    if model_type.lower() == 'xception':
        transform = XCEPTION_TRANSFORM
    elif model_type.lower() == 'efficientnet':
        transform = EFFICIENTNET_TRANSFORM
    elif model_type.lower() == 'mesonet':
        transform = MESONET_TRANSFORM
    else:
        transform = XCEPTION_TRANSFORM  # Default
    
    # Apply transformation
    tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return tensor


def extract_face_landmarks(image_path: str, use_dlib: bool = False) -> List[Dict[str, Any]]:
    """
    Extract facial landmarks using either dlib or OpenCV.
    
    Args:
        image_path: Path to the image file
        use_dlib: Whether to use dlib (if available) or OpenCV
        
    Returns:
        List of dictionaries containing landmarks for each detected face
    """
    if use_dlib:
        try:
            import dlib
            
            # Load the image
            img = cv2.imread(image_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Load the detector and predictor
            detector = dlib.get_frontal_face_detector()
            
            # Check if shape predictor model exists
            predictor_path = os.path.join("models", "shape_predictor_68_face_landmarks.dat")
            if not os.path.exists(predictor_path):
                logger.warning(f"Shape predictor model not found at {predictor_path}")
                return []
                
            predictor = dlib.shape_predictor(predictor_path)
            
            # Detect faces
            faces = detector(gray)
            
            landmarks_list = []
            for face in faces:
                # Get landmarks
                landmarks = predictor(gray, face)
                
                # Convert to dictionary
                points = {}
                for i in range(68):
                    point = landmarks.part(i)
                    points[i] = (point.x, point.y)
                
                landmarks_list.append(points)
            
            return landmarks_list
        except ImportError:
            logger.warning("dlib not available, falling back to OpenCV")
    
    # Fallback to OpenCV
    try:
        # Load the image
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        landmarks_list = []
        for (x, y, w, h) in faces:
            # Create a simple landmark dictionary with face rectangle corners
            landmarks = {
                "face_rect": (x, y, w, h),
                "left_eye": (x + w//4, y + h//3),
                "right_eye": (x + 3*w//4, y + h//3),
                "nose": (x + w//2, y + h//2),
                "mouth_left": (x + w//3, y + 2*h//3),
                "mouth_right": (x + 2*w//3, y + 2*h//3)
            }
            landmarks_list.append(landmarks)
        
        return landmarks_list
    except Exception as e:
        logger.error(f"Error extracting landmarks: {str(e)}")
        return []


def analyze_image_frequency(image_path: str) -> Dict[str, float]:
    """
    Analyze frequency components of an image.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Dictionary with frequency analysis results
    """
    try:
        # Load the image in grayscale
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Apply FFT
        f_transform = np.fft.fft2(image)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
        
        # Calculate statistics
        low_freq_mask = np.zeros_like(magnitude_spectrum, dtype=bool)
        mid_freq_mask = np.zeros_like(magnitude_spectrum, dtype=bool)
        high_freq_mask = np.zeros_like(magnitude_spectrum, dtype=bool)
        
        rows, cols = magnitude_spectrum.shape
        center_r, center_c = rows // 2, cols // 2
        
        for r in range(rows):
            for c in range(cols):
                dist = np.sqrt((r - center_r) ** 2 + (c - center_c) ** 2)
                if dist < min(rows, cols) * 0.1:
                    low_freq_mask[r, c] = True
                elif dist < min(rows, cols) * 0.3:
                    mid_freq_mask[r, c] = True
                else:
                    high_freq_mask[r, c] = True
        
        # Calculate energy in each frequency band
        low_freq_energy = np.sum(magnitude_spectrum[low_freq_mask]) / np.sum(low_freq_mask)
        mid_freq_energy = np.sum(magnitude_spectrum[mid_freq_mask]) / np.sum(mid_freq_mask)
        high_freq_energy = np.sum(magnitude_spectrum[high_freq_mask]) / np.sum(high_freq_mask)
        
        # Calculate high-to-low frequency ratio
        high_to_low_ratio = high_freq_energy / low_freq_energy if low_freq_energy > 0 else 0
        
        return {
            "low_freq_energy": float(low_freq_energy),
            "mid_freq_energy": float(mid_freq_energy),
            "high_freq_energy": float(high_freq_energy),
            "high_to_low_ratio": float(high_to_low_ratio)
        }
    except Exception as e:
        logger.error(f"Error in frequency analysis: {str(e)}")
        return {
            "error": str(e),
            "low_freq_energy": 0.0,
            "mid_freq_energy": 0.0,
            "high_freq_energy": 0.0,
            "high_to_low_ratio": 0.0
        }


def extract_image_features(image_path: str) -> Dict[str, Any]:
    """
    Extract various features from an image.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Dictionary with extracted features
    """
    try:
        # Load the image
        img = cv2.imread(image_path)
        
        # Convert to different color spaces
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Calculate basic statistics
        gray_mean = np.mean(gray)
        gray_std = np.std(gray)
        gray_median = np.median(gray)
        
        # Calculate histogram
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()  # Normalize
        
        # Calculate texture features using GLCM
        from skimage.feature import graycomatrix, graycoprops
        
        glcm = graycomatrix(
            gray, 
            distances=[1], 
            angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], 
            levels=256,
            symmetric=True, 
            normed=True
        )
        
        contrast = graycoprops(glcm, 'contrast').mean()
        dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
        homogeneity = graycoprops(glcm, 'homogeneity').mean()
        energy = graycoprops(glcm, 'energy').mean()
        correlation = graycoprops(glcm, 'correlation').mean()
        
        # Calculate edge features
        edges = cv2.Canny(gray, 100, 200)
        edge_ratio = np.count_nonzero(edges) / (edges.shape[0] * edges.shape[1])
        
        return {
            "basic_stats": {
                "mean": float(gray_mean),
                "std": float(gray_std),
                "median": float(gray_median)
            },
            "histogram": hist.tolist(),
            "texture": {
                "contrast": float(contrast),
                "dissimilarity": float(dissimilarity),
                "homogeneity": float(homogeneity),
                "energy": float(energy),
                "correlation": float(correlation)
            },
            "edge_ratio": float(edge_ratio)
        }
    except Exception as e:
        logger.error(f"Error extracting image features: {str(e)}")
        return {"error": str(e)}