import os
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List, Any, Optional
import logging
import uuid

from app.core.config import settings

# Configure logging
logger = logging.getLogger(__name__)


def generate_heatmap_visualization(
    image: np.ndarray, 
    heatmap: np.ndarray, 
    output_path: Optional[str] = None
) -> str:
    """
    Generate a visualization of a heatmap overlaid on an image.
    
    Args:
        image (np.ndarray): The original image
        heatmap (np.ndarray): The heatmap
        output_path (str, optional): Path to save the visualization
    
    Returns:
        str: Path to the saved visualization
    """
    try:
        if output_path is None:
            # Create a unique filename
            filename = f"heatmap_{uuid.uuid4().hex}.png"
            output_path = os.path.join("visualizations", filename)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Resize heatmap to match image if needed
        if heatmap.shape[:2] != image.shape[:2]:
            heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        
        # Normalize heatmap
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        # Convert to colormap
        heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        # Convert image to BGR if it's RGB
        if image.shape[2] == 3 and image[0, 0, 0] <= image[0, 0, 2]:  # Simple RGB check
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image
        
        # Blend image and heatmap
        alpha = 0.6
        overlay = cv2.addWeighted(image_bgr, 1 - alpha, heatmap_colored, alpha, 0)
        
        # Save visualization
        cv2.imwrite(output_path, overlay)
        
        return output_path
    
    except Exception as e:
        logger.error(f"Error generating heatmap visualization: {str(e)}")
        return ""


def generate_temporal_visualization(
    timestamps: List[float], 
    values: List[float], 
    threshold: float = 0.5, 
    title: str = "Temporal Analysis", 
    output_path: Optional[str] = None
) -> str:
    """
    Generate a visualization of temporal analysis.
    
    Args:
        timestamps (List[float]): Timestamps
        values (List[float]): Values at each timestamp
        threshold (float): Threshold line
        title (str): Title for the visualization
        output_path (str, optional): Path to save the visualization
    
    Returns:
        str: Path to the saved visualization
    """
    try:
        if output_path is None:
            # Create a unique filename
            filename = f"temporal_{uuid.uuid4().hex}.png"
            output_path = os.path.join("visualizations", filename)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Create figure
        plt.figure(figsize=(10, 6))
        plt.plot(timestamps, values, 'b-', linewidth=2)
        plt.axhline(y=threshold, color='r', linestyle='--', alpha=0.7)
        
        # Add labels and title
        plt.xlabel("Time (s)")
        plt.ylabel("Score")
        plt.title(title)
        
        # Add grid and legend
        plt.grid(True, alpha=0.3)
        plt.legend(["Detection Score", "Threshold"])
        
        # Customize appearance
        plt.tight_layout()
        
        # Save figure
        plt.savefig(output_path, dpi=100)
        plt.close()
        
        return output_path
    
    except Exception as e:
        logger.error(f"Error generating temporal visualization: {str(e)}")
        return ""


def generate_frequency_visualization(
    frequency_data: Dict[str, float], 
    title: str = "Frequency Analysis", 
    output_path: Optional[str] = None
) -> str:
    """
    Generate a visualization of frequency analysis.
    
    Args:
        frequency_data (Dict[str, float]): Frequency analysis data
        title (str): Title for the visualization
        output_path (str, optional): Path to save the visualization
    
    Returns:
        str: Path to the saved visualization
    """
    try:
        if output_path is None:
            # Create a unique filename
            filename = f"frequency_{uuid.uuid4().hex}.png"
            output_path = os.path.join("visualizations", filename)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Extract data
        categories = list(frequency_data.keys())
        values = list(frequency_data.values())
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Bar plot
        bars = plt.bar(range(len(categories)), values, color='skyblue')
        
        # Add labels and title
        plt.xlabel("Frequency Band")
        plt.ylabel("Energy")
        plt.title(title)
        plt.xticks(range(len(categories)), categories, rotation=45)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2f}', ha='center', va='bottom')
        
        # Customize appearance
        plt.tight_layout()
        
        # Save figure
        plt.savefig(output_path, dpi=100)
        plt.close()
        
        return output_path
    
    except Exception as e:
        logger.error(f"Error generating frequency visualization: {str(e)}")
        return ""


def generate_confidence_gauge(
    confidence: float, 
    output_path: Optional[str] = None
) -> str:
    """
    Generate a gauge visualization for confidence score.
    
    Args:
        confidence (float): Confidence score (0-1)
        output_path (str, optional): Path to save the visualization
    
    Returns:
        str: Path to the saved visualization
    """
    try:
        if output_path is None:
            # Create a unique filename
            filename = f"gauge_{uuid.uuid4().hex}.png"
            output_path = os.path.join("visualizations", filename)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Set up the gauge figure
        fig = plt.figure(figsize=(8, 4))
        ax = fig.add_subplot(111)
        
        # Hide regular axes
        ax.set_axis_off()
        
        # Create gauge background
        gauge_background = plt.Rectangle((0, 0), 1, 0.3, facecolor='lightgray', alpha=0.3)
        ax.add_patch(gauge_background)
        
        # Create gauge fill
        gauge_fill = plt.Rectangle((0, 0), confidence, 0.3, facecolor='red' if confidence > 0.5 else 'green')
        ax.add_patch(gauge_fill)
        
        # Add threshold marker
        plt.axvline(x=0.5, color='black', linestyle='--', alpha=0.7, linewidth=2)
        
        # Add labels
        plt.text(0.05, 0.4, "Real", fontsize=12)
        plt.text(0.85, 0.4, "Fake", fontsize=12)
        plt.text(confidence, 0.15, f"{confidence:.2f}", fontsize=14, 
                 horizontalalignment='center', verticalalignment='center',
                 color='white' if 0.3 <= confidence <= 0.7 else 'black',
                 weight='bold')
        
        # Set axis limits
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 0.5)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(output_path, dpi=100)
        plt.close()
        
        return output_path
    
    except Exception as e:
        logger.error(f"Error generating confidence gauge: {str(e)}")
        return ""