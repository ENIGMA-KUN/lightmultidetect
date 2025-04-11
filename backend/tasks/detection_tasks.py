import os
import time
import json
import numpy as np
import torch
from typing import List, Dict, Any, Optional
import logging
from celery import Celery
from celery.utils.log import get_task_logger
import matplotlib.pyplot as plt
import cv2
from PIL import Image

from backend.core.config import settings

# Initialize Celery
celery_app = Celery(
    "detection_tasks",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND
)

# Configure Celery
celery_app.conf.task_routes = {
    "backend.tasks.detection_tasks.*": {"queue": "detection"}
}

# Configure logging
logger = get_task_logger(__name__)


@celery_app.task(bind=True, name="process_image")
def process_image(
    self,
    task_id: str,
    image_files: List[str],
    confidence_threshold: float = 0.5,
    explain_results: bool = True
) -> Dict[str, Any]:
    """
    Process images for deepfake detection
    
    Args:
        task_id: Task ID
        image_files: List of image file paths
        confidence_threshold: Confidence threshold for detection
        explain_results: Whether to include explanations in results
    
    Returns:
        Detection results
    """
    logger.info(f"Processing {len(image_files)} images for task {task_id}")
    
    try:
        # Create results directory
        results_dir = os.path.join(settings.RESULTS_DIR, task_id)
        os.makedirs(results_dir, exist_ok=True)
        
        # Create visualizations directory
        viz_dir = os.path.join(results_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Load model
        model_path = os.path.join(settings.MODEL_DIR, "image_model.pth")
        model = load_image_model(model_path)
        
        # Process each image
        results = []
        
        for i, image_file in enumerate(image_files):
            # Update progress
            self.update_state(
                state="PROGRESS",
                meta={"current": i, "total": len(image_files)}
            )
            
            # Process image
            start_time = time.time()
            result = process_single_image(
                image_file=image_file,
                model=model,
                confidence_threshold=confidence_threshold,
                explain=explain_results,
                viz_dir=viz_dir
            )
            end_time = time.time()
            
            # Add processing time
            result["processing_time"] = end_time - start_time
            
            # Add to results
            results.append(result)
        
        # Save results
        result_data = {
            "task_id": task_id,
            "modality": "image",
            "num_files": len(image_files),
            "results": results,
            "completed_at": time.time()
        }
        
        with open(os.path.join(results_dir, "result.json"), "w") as f:
            json.dump(result_data, f, indent=2)
        
        logger.info(f"Image processing completed for task {task_id}")
        
        return result_data
        
    except Exception as e:
        logger.error(f"Error processing images for task {task_id}: {e}")
        
        # Create error file
        with open(os.path.join(results_dir, "error.txt"), "w") as f:
            f.write(str(e))
        
        raise


@celery_app.task(bind=True, name="process_audio")
def process_audio(
    self,
    task_id: str,
    audio_files: List[str],
    confidence_threshold: float = 0.5,
    explain_results: bool = True
) -> Dict[str, Any]:
    """
    Process audio for deepfake detection
    
    Args:
        task_id: Task ID
        audio_files: List of audio file paths
        confidence_threshold: Confidence threshold for detection
        explain_results: Whether to include explanations in results
    
    Returns:
        Detection results
    """
    logger.info(f"Processing {len(audio_files)} audio files for task {task_id}")
    
    try:
        # Create results directory
        results_dir = os.path.join(settings.RESULTS_DIR, task_id)
        os.makedirs(results_dir, exist_ok=True)
        
        # Create visualizations directory
        viz_dir = os.path.join(results_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Load model
        model_path = os.path.join(settings.MODEL_DIR, "audio_model.pth")
        model = load_audio_model(model_path)
        
        # Process each audio file
        results = []
        
        for i, audio_file in enumerate(audio_files):
            # Update progress
            self.update_state(
                state="PROGRESS",
                meta={"current": i, "total": len(audio_files)}
            )
            
            # Process audio
            start_time = time.time()
            result = process_single_audio(
                audio_file=audio_file,
                model=model,
                confidence_threshold=confidence_threshold,
                explain=explain_results,
                viz_dir=viz_dir
            )
            end_time = time.time()
            
            # Add processing time
            result["processing_time"] = end_time - start_time
            
            # Add to results
            results.append(result)
        
        # Save results
        result_data = {
            "task_id": task_id,
            "modality": "audio",
            "num_files": len(audio_files),
            "results": results,
            "completed_at": time.time()
        }
        
        with open(os.path.join(results_dir, "result.json"), "w") as f:
            json.dump(result_data, f, indent=2)
        
        logger.info(f"Audio processing completed for task {task_id}")
        
        return result_data
        
    except Exception as e:
        logger.error(f"Error processing audio for task {task_id}: {e}")
        
        # Create error file
        with open(os.path.join(results_dir, "error.txt"), "w") as f:
            f.write(str(e))
        
        raise


@celery_app.task(bind=True, name="process_video")
def process_video(
    self,
    task_id: str,
    video_files: List[str],
    confidence_threshold: float = 0.5,
    explain_results: bool = True
) -> Dict[str, Any]:
    """
    Process videos for deepfake detection
    
    Args:
        task_id: Task ID
        video_files: List of video file paths
        confidence_threshold: Confidence threshold for detection
        explain_results: Whether to include explanations in results
    
    Returns:
        Detection results
    """
    logger.info(f"Processing {len(video_files)} videos for task {task_id}")
    
    try:
        # Create results directory
        results_dir = os.path.join(settings.RESULTS_DIR, task_id)
        os.makedirs(results_dir, exist_ok=True)
        
        # Create visualizations directory
        viz_dir = os.path.join(results_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Load model
        model_path = os.path.join(settings.MODEL_DIR, "video_model.pth")
        model = load_video_model(model_path)
        
        # Process each video
        results = []
        
        for i, video_file in enumerate(video_files):
            # Update progress
            self.update_state(
                state="PROGRESS",
                meta={"current": i, "total": len(video_files)}
            )
            
            # Process video
            start_time = time.time()
            result = process_single_video(
                video_file=video_file,
                model=model,
                confidence_threshold=confidence_threshold,
                explain=explain_results,
                viz_dir=viz_dir
            )
            end_time = time.time()
            
            # Add processing time
            result["processing_time"] = end_time - start_time
            
            # Add to results
            results.append(result)
        
        # Save results
        result_data = {
            "task_id": task_id,
            "modality": "video",
            "num_files": len(video_files),
            "results": results,
            "completed_at": time.time()
        }
        
        with open(os.path.join(results_dir, "result.json"), "w") as f:
            json.dump(result_data, f, indent=2)
        
        logger.info(f"Video processing completed for task {task_id}")
        
        return result_data
        
    except Exception as e:
        logger.error(f"Error processing videos for task {task_id}: {e}")
        
        # Create error file
        with open(os.path.join(results_dir, "error.txt"), "w") as f:
            f.write(str(e))
        
        raise


@celery_app.task(bind=True, name="process_multimodal")
def process_multimodal(
    self,
    task_id: str,
    image_files: List[str],
    audio_files: List[str],
    video_files: List[str],
    confidence_threshold: float = 0.5,
    explain_results: bool = True
) -> Dict[str, Any]:
    """
    Process media with multi-modal deepfake detection
    
    Args:
        task_id: Task ID
        image_files: List of image file paths
        audio_files: List of audio file paths
        video_files: List of video file paths
        confidence_threshold: Confidence threshold for detection
        explain_results: Whether to include explanations in results
    
    Returns:
        Detection results
    """
    logger.info(f"Processing multi-modal data for task {task_id}")
    
    try:
        # Create results directory
        results_dir = os.path.join(settings.RESULTS_DIR, task_id)
        os.makedirs(results_dir, exist_ok=True)
        
        # Create visualizations directory
        viz_dir = os.path.join(results_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Load multi-modal model
        model_path = os.path.join(settings.MODEL_DIR, "multimodal_model.pth")
        model = load_multimodal_model(model_path)
        
        # Process each media type
        all_results = []
        
        # Process images
        if image_files:
            for i, image_file in enumerate(image_files):
                # Update progress
                self.update_state(
                    state="PROGRESS",
                    meta={
                        "current": i,
                        "total": len(image_files) + len(audio_files) + len(video_files),
                        "type": "image"
                    }
                )
                
                # Process image
                start_time = time.time()
                result = process_single_image(
                    image_file=image_file,
                    model=model,
                    confidence_threshold=confidence_threshold,
                    explain=explain_results,
                    viz_dir=viz_dir
                )
                end_time = time.time()
                
                # Add processing time
                result["processing_time"] = end_time - start_time
                
                # Add to results
                all_results.append(result)
        
        # Process audio files
        if audio_files:
            for i, audio_file in enumerate(audio_files):
                # Update progress
                self.update_state(
                    state="PROGRESS",
                    meta={
                        "current": len(image_files) + i,
                        "total": len(image_files) + len(audio_files) + len(video_files),
                        "type": "audio"
                    }
                )
                
                # Process audio
                start_time = time.time()
                result = process_single_audio(
                    audio_file=audio_file,
                    model=model,
                    confidence_threshold=confidence_threshold,
                    explain=explain_results,
                    viz_dir=viz_dir
                )
                end_time = time.time()
                
                # Add processing time
                result["processing_time"] = end_time - start_time
                
                # Add to results
                all_results.append(result)
        
        # Process video files
        if video_files:
            for i, video_file in enumerate(video_files):
                # Update progress
                self.update_state(
                    state="PROGRESS",
                    meta={
                        "current": len(image_files) + len(audio_files) + i,
                        "total": len(image_files) + len(audio_files) + len(video_files),
                        "type": "video"
                    }
                )
                
                # Process video
                start_time = time.time()
                result = process_single_video(
                    video_file=video_file,
                    model=model,
                    confidence_threshold=confidence_threshold,
                    explain=explain_results,
                    viz_dir=viz_dir
                )
                end_time = time.time()
                
                # Add processing time
                result["processing_time"] = end_time - start_time
                
                # Add to results
                all_results.append(result)
        
        # Save results
        result_data = {
            "task_id": task_id,
            "modality": "multimodal",
            "num_files": {
                "image": len(image_files),
                "audio": len(audio_files),
                "video": len(video_files),
                "total": len(image_files) + len(audio_files) + len(video_files)
            },
            "results": all_results,
            "completed_at": time.time()
        }
        
        with open(os.path.join(results_dir, "result.json"), "w") as f:
            json.dump(result_data, f, indent=2)
        
        logger.info(f"Multi-modal processing completed for task {task_id}")
        
        return result_data
        
    except Exception as e:
        logger.error(f"Error processing multi-modal data for task {task_id}: {e}")
        
        # Create error file
        with open(os.path.join(results_dir, "error.txt"), "w") as f:
            f.write(str(e))
        
        raise


# Helper functions (to be implemented with actual model loading and inference)
def load_image_model(model_path: str):
    """Load image model (placeholder)"""
    # This would be implemented to load the actual model
    return "image_model"


def load_audio_model(model_path: str):
    """Load audio model (placeholder)"""
    # This would be implemented to load the actual model
    return "audio_model"


def load_video_model(model_path: str):
    """Load video model (placeholder)"""
    # This would be implemented to load the actual model
    return "video_model"


def load_multimodal_model(model_path: str):
    """Load multi-modal model (placeholder)"""
    # This would be implemented to load the actual model
    return "multimodal_model"


def process_single_image(
    image_file: str,
    model,
    confidence_threshold: float,
    explain: bool,
    viz_dir: str
) -> Dict[str, Any]:
    """
    Process a single image for deepfake detection (placeholder)
    
    In a real implementation, this would:
    1. Load the image
    2. Preprocess it
    3. Run inference with the model
    4. Generate explanations if requested
    5. Create visualizations
    6. Return results
    """
    # Simulate processing delay
    time.sleep(0.5)
    
    # Create a random prediction for demonstration
    is_fake = np.random.random() > 0.5
    confidence = np.random.uniform(0.7, 0.95)
    
    # File name without path
    file_name = os.path.basename(image_file)
    
    # Create visualization
    viz_file = None
    if explain:
        viz_file = f"{os.path.splitext(file_name)[0]}_viz.png"
        viz_path = os.path.join(viz_dir, viz_file)
        
        # Create a simple visualization (placeholder)
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        img = plt.imread(image_file)
        plt.imshow(img)
        plt.title("Original Image")
        
        plt.subplot(1, 2, 2)
        # Generate a heatmap (placeholder)
        heatmap = np.random.rand(img.shape[0], img.shape[1])
        plt.imshow(img)
        plt.imshow(heatmap, cmap='jet', alpha=0.5)
        plt.title("Deepfake Detection Heatmap")
        
        plt.suptitle(f"Prediction: {'Fake' if is_fake else 'Real'} ({confidence:.2f})")
        plt.tight_layout()
        plt.savefig(viz_path)
        plt.close()
    
    # Create explanation
    explanation = None
    if explain:
        explanation = {
            "important_regions": [
                {
                    "x": int(np.random.uniform(0, 1) * 100),
                    "y": int(np.random.uniform(0, 1) * 100),
                    "width": int(np.random.uniform(20, 50)),
                    "height": int(np.random.uniform(20, 50)),
                    "importance": np.random.uniform(0.7, 0.9)
                }
            ],
            "frequency_analysis": {
                "low_freq_energy": np.random.uniform(0.2, 0.4),
                "mid_freq_energy": np.random.uniform(0.3, 0.5),
                "high_freq_energy": np.random.uniform(0.1, 0.3)
            }
        }
    
    return {
        "file_name": file_name,
        "predicted_label": "fake" if is_fake else "real",
        "confidence": float(confidence),
        "modality": "image",
        "visualization_file": viz_file,
        "explanation": explanation
    }


def process_single_audio(
    audio_file: str,
    model,
    confidence_threshold: float,
    explain: bool,
    viz_dir: str
) -> Dict[str, Any]:
    """
    Process a single audio file for deepfake detection (placeholder)
    
    In a real implementation, this would:
    1. Load the audio
    2. Preprocess it
    3. Run inference with the model
    4. Generate explanations if requested
    5. Create visualizations
    6. Return results
    """
    # Simulate processing delay
    time.sleep(0.7)
    
    # Create a random prediction for demonstration
    is_fake = np.random.random() > 0.5
    confidence = np.random.uniform(0.7, 0.95)
    
    # File name without path
    file_name = os.path.basename(audio_file)
    
    # Create visualization
    viz_file = None
    if explain:
        viz_file = f"{os.path.splitext(file_name)[0]}_viz.png"
        viz_path = os.path.join(viz_dir, viz_file)
        
        # Create a simple visualization (placeholder)
        plt.figure(figsize=(10, 6))
        
        # Generate a spectrogram (placeholder)
        t = np.linspace(0, 10, 1000)
        freq = np.linspace(0, 1000, 200)
        spectrogram = np.random.rand(len(freq), len(t))
        
        plt.imshow(spectrogram, aspect='auto', origin='lower', extent=[0, 10, 0, 1000])
        plt.colorbar(label='Magnitude')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.title(f"Audio Spectrogram - Prediction: {'Fake' if is_fake else 'Real'} ({confidence:.2f})")
        plt.tight_layout()
        plt.savefig(viz_path)
        plt.close()
    
    # Create explanation
    explanation = None
    if explain:
        explanation = {
            "temporal_consistency": np.random.uniform(0.5, 0.9),
            "spectral_artifacts": {
                "presence": np.random.choice([True, False]),
                "strength": np.random.uniform(0.1, 0.8)
            },
            "frequency_bands": {
                "low": np.random.uniform(0.2, 0.4),
                "mid": np.random.uniform(0.3, 0.5),
                "high": np.random.uniform(0.1, 0.3)
            }
        }
    
    return {
        "file_name": file_name,
        "predicted_label": "fake" if is_fake else "real",
        "confidence": float(confidence),
        "modality": "audio",
        "visualization_file": viz_file,
        "explanation": explanation
    }


def process_single_video(
    video_file: str,
    model,
    confidence_threshold: float,
    explain: bool,
    viz_dir: str
) -> Dict[str, Any]:
    """
    Process a single video for deepfake detection (placeholder)
    
    In a real implementation, this would:
    1. Load the video
    2. Extract frames and audio
    3. Preprocess them
    4. Run inference with the model
    5. Generate explanations if requested
    6. Create visualizations
    7. Return results
    """
    # Simulate processing delay
    time.sleep(1.5)
    
    # Create a random prediction for demonstration
    is_fake = np.random.random() > 0.5
    confidence = np.random.uniform(0.7, 0.95)
    
    # File name without path
    file_name = os.path.basename(video_file)
    
    # Create visualization
    viz_file = None
    if explain:
        viz_file = f"{os.path.splitext(file_name)[0]}_viz.png"
        viz_path = os.path.join(viz_dir, viz_file)
        
        # Create a simple visualization (placeholder)
        plt.figure(figsize=(15, 8))
        
        # Simulate video frames
        n_frames = 6
        for i in range(n_frames):
            plt.subplot(2, 3, i+1)
            
            # Generate a random frame (placeholder)
            frame = np.random.rand(100, 100, 3)
            plt.imshow(frame)
            plt.title(f"Frame {i+1}")
            plt.axis('off')
        
        plt.suptitle(f"Video Analysis - Prediction: {'Fake' if is_fake else 'Real'} ({confidence:.2f})")
        plt.tight_layout()
        plt.savefig(viz_path)
        plt.close()
    
    # Create explanation
    explanation = None
    if explain:
        explanation = {
            "temporal_consistency": np.random.uniform(0.5, 0.9),
            "facial_landmarks": {
                "stability": np.random.uniform(0.6, 0.95),
                "abnormalities": np.random.choice([True, False])
            },
            "audio_visual_sync": np.random.uniform(0.7, 0.95),
            "important_frames": [
                {
                    "frame_idx": int(np.random.uniform(0, 100)),
                    "confidence": np.random.uniform(0.7, 0.9)
                },
                {
                    "frame_idx": int(np.random.uniform(100, 200)),
                    "confidence": np.random.uniform(0.7, 0.9)
                }
            ]
        }
    
    return {
        "file_name": file_name,
        "predicted_label": "fake" if is_fake else "real",
        "confidence": float(confidence),
        "modality": "video",
        "visualization_file": viz_file,
        "explanation": explanation
    } 