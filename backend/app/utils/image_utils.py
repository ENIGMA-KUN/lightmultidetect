import os
import cv2
import numpy as np
import PIL.Image
from typing import List, Tuple, Optional, Union
import logging

# Configure logging
logger = logging.getLogger(__name__)


def is_valid_image(image_path: str) -> bool:
    """
    Check if the file is a valid image.
    
    Args:
        image_path (str): Path to the image file
    
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        img = PIL.Image.open(image_path)
        img.verify()  # Verify it's an image
        return True
    except Exception as e:
        logger.error(f"Invalid image file: {str(e)}")
        return False


def detect_faces(image_path: str) -> List[np.ndarray]:
    """
    Detect faces in an image.
    
    Args:
        image_path (str): Path to the image file
    
    Returns:
        List[np.ndarray]: List of detected face regions
    """
    try:
        # Load the image
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Load the face detector
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        # Extract face regions
        face_regions = []
        for (x, y, w, h) in faces:
            face_regions.append(image[y:y+h, x:x+w])
        
        return face_regions
    
    except Exception as e:
        logger.error(f"Error detecting faces: {str(e)}")
        return []


def extract_facial_landmarks(image_path: str) -> List[dict]:
    """
    Extract facial landmarks using dlib (if available).
    
    Args:
        image_path (str): Path to the image file
    
    Returns:
        List[dict]: List of landmarks for each detected face
    """
    try:
        import dlib
        
        # Load the image
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Load the face detector and landmark predictor
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        
        # Detect faces
        faces = detector(gray)
        
        landmarks_list = []
        for face in faces:
            # Get landmarks
            landmarks = predictor(gray, face)
            
            # Convert to more usable format
            points = {}
            for i in range(68):
                point = landmarks.part(i)
                points[i] = (point.x, point.y)
            
            landmarks_list.append(points)
        
        return landmarks_list
    
    except ImportError:
        logger.warning("dlib not available for facial landmark extraction")
        return []
    except Exception as e:
        logger.error(f"Error extracting facial landmarks: {str(e)}")
        return []


def analyze_image_frequencies(image_path: str) -> dict:
    """
    Analyze frequency components of an image.
    
    Args:
        image_path (str): Path to the image file
    
    Returns:
        dict: Frequency analysis results
    """
    try:
        # Load the image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
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
        
        low_freq_energy = np.sum(magnitude_spectrum[low_freq_mask]) / np.sum(low_freq_mask)
        mid_freq_energy = np.sum(magnitude_spectrum[mid_freq_mask]) / np.sum(mid_freq_mask)
        high_freq_energy = np.sum(magnitude_spectrum[high_freq_mask]) / np.sum(high_freq_mask)
        
        # GAN-generated images often have abnormal high frequency patterns
        # Calculate high-to-low frequency ratio
        high_to_low_ratio = high_freq_energy / low_freq_energy if low_freq_energy > 0 else 0
        
        return {
            "low_freq_energy": float(low_freq_energy),
            "mid_freq_energy": float(mid_freq_energy),
            "high_freq_energy": float(high_freq_energy),
            "high_to_low_ratio": float(high_to_low_ratio)
        }
    
    except Exception as e:
        logger.error(f"Error analyzing image frequencies: {str(e)}")
        return {
            "error": str(e),
            "low_freq_energy": 0.0,
            "mid_freq_energy": 0.0,
            "high_freq_energy": 0.0,
            "high_to_low_ratio": 0.0
        }