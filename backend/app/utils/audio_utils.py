import os
import numpy as np
import librosa
import soundfile as sf
from typing import Tuple, Optional
import logging

# Configure logging
logger = logging.getLogger(__name__)


def is_valid_audio(audio_path: str) -> bool:
    """
    Check if the file is a valid audio file.
    
    Args:
        audio_path (str): Path to the audio file
    
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        y, sr = librosa.load(audio_path, sr=None, duration=1.0)  # Just load a small portion
        return True
    except Exception as e:
        logger.error(f"Invalid audio file: {str(e)}")
        return False


def extract_audio_from_video(video_path: str, output_path: Optional[str] = None) -> str:
    """
    Extract audio track from a video file.
    
    Args:
        video_path (str): Path to the video file
        output_path (str, optional): Path to save the extracted audio
    
    Returns:
        str: Path to the extracted audio file
    """
    try:
        import subprocess
        
        if output_path is None:
            # Create temporary file
            output_path = video_path + ".wav"
        
        # Use ffmpeg to extract audio
        cmd = [
            "ffmpeg",
            "-i", video_path,
            "-q:a", "0",
            "-map", "a",
            output_path,
            "-y"  # Overwrite existing file if any
        ]
        
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        return output_path
    
    except Exception as e:
        logger.error(f"Error extracting audio from video: {str(e)}")
        return ""


def analyze_audio_features(audio_path: str) -> dict:
    """
    Extract and analyze audio features.
    
    Args:
        audio_path (str): Path to the audio file
    
    Returns:
        dict: Audio feature analysis
    """
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=16000)
        
        # Extract various features
        # MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs, axis=1)
        
        # Spectral contrast
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        contrast_mean = np.mean(contrast, axis=1)
        
        # Chroma features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = np.mean(zcr)
        
        # Spectral rolloff
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        rolloff_mean = np.mean(rolloff)
        
        # Spectral flatness
        flatness = librosa.feature.spectral_flatness(y=y)
        flatness_mean = np.mean(flatness)
        
        return {
            "mfccs": mfccs_mean.tolist(),
            "spectral_contrast": contrast_mean.tolist(),
            "chroma": chroma_mean.tolist(),
            "zero_crossing_rate": float(zcr_mean),
            "spectral_rolloff": float(rolloff_mean),
            "spectral_flatness": float(flatness_mean)
        }
    
    except Exception as e:
        logger.error(f"Error analyzing audio features: {str(e)}")
        return {
            "error": str(e)
        }


def check_voice_consistency(audio_path: str) -> dict:
    """
    Check for voice consistency in audio (useful for deepfake detection).
    
    Args:
        audio_path (str): Path to the audio file
    
    Returns:
        dict: Voice consistency analysis
    """
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=16000)
        
        # Extract features over time
        # Segment audio into chunks
        segment_length = sr * 1  # 1 second segments
        num_segments = len(y) // segment_length
        
        if num_segments < 2:
            return {
                "consistency_score": 1.0,
                "segments_analyzed": 1,
                "error": "Audio too short for consistency analysis"
            }
        
        # Extract MFCCs for each segment
        segment_mfccs = []
        for i in range(num_segments):
            segment = y[i * segment_length:(i + 1) * segment_length]
            mfccs = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=13)
            segment_mfccs.append(np.mean(mfccs, axis=1))
        
        # Calculate variance between segments
        segment_mfccs = np.array(segment_mfccs)
        variance = np.mean(np.var(segment_mfccs, axis=0))
        
        # Calculate consistency score (inverse of variance, normalized)
        consistency_score = 1.0 / (1.0 + variance)
        
        return {
            "consistency_score": float(consistency_score),
            "segments_analyzed": num_segments,
            "variance": float(variance)
        }
    
    except Exception as e:
        logger.error(f"Error checking voice consistency: {str(e)}")
        return {
            "consistency_score": 0.0,
            "segments_analyzed": 0,
            "error": str(e)
        }