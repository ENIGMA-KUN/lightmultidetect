import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import logging
from typing import Dict, Tuple, Any, List, Union, Optional

from app.core.config import settings
from app.utils.video_utils import extract_frames, extract_faces

# Configure logging
logger = logging.getLogger(__name__)

# Global model cache
_video_models = {}


class C3D(nn.Module):
    """3D Convolutional Network for video analysis."""
    
    def __init__(self, num_classes=1):
        super(C3D, self).__init__()
        
        # 3D convolutional layers
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        
        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))
        
        # Fully connected layers
        self.fc6 = nn.Linear(8192, 4096)
        self.fc7 = nn.Linear(4096, 1024)
        self.fc8 = nn.Linear(1024, num_classes)
        
        # Dropout layers
        self.dropout = nn.Dropout(p=0.5)
        
        # Activation for binary classification
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        h = self.pool1(F.relu(self.conv1(x)))
        h = self.pool2(F.relu(self.conv2(h)))
        
        h = F.relu(self.conv3a(h))
        h = self.pool3(F.relu(self.conv3b(h)))
        
        h = F.relu(self.conv4a(h))
        h = self.pool4(F.relu(self.conv4b(h)))
        
        h = F.relu(self.conv5a(h))
        h = self.pool5(F.relu(self.conv5b(h)))
        
        h = h.view(-1, 8192)
        h = self.dropout(F.relu(self.fc6(h)))
        h = self.dropout(F.relu(self.fc7(h)))
        h = self.fc8(h)
        
        return self.sigmoid(h)
    
    def get_features(self, x):
        """Extract features before final classification."""
        h = self.pool1(F.relu(self.conv1(x)))
        h = self.pool2(F.relu(self.conv2(h)))
        
        h = F.relu(self.conv3a(h))
        h = self.pool3(F.relu(self.conv3b(h)))
        
        h = F.relu(self.conv4a(h))
        h = self.pool4(F.relu(self.conv4b(h)))
        
        h = F.relu(self.conv5a(h))
        h = self.pool5(F.relu(self.conv5b(h)))
        
        h = h.view(-1, 8192)
        h = self.dropout(F.relu(self.fc6(h)))
        h = self.dropout(F.relu(self.fc7(h)))
        
        return h


class TwoStreamNetwork(nn.Module):
    """Two-stream network for video deepfake detection."""
    
    def __init__(self, num_classes=1):
        super(TwoStreamNetwork, self).__init__()
        
        # Spatial stream (based on ResNet-18)
        import torchvision.models as models
        self.spatial_stream = models.resnet18(pretrained=True)
        num_ftrs_spatial = self.spatial_stream.fc.in_features
        self.spatial_stream.fc = nn.Identity()  # Remove the final FC layer
        
        # Temporal stream (optical flow)
        self.temporal_stream = models.resnet18(pretrained=True)
        self.temporal_stream.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)  # Modify for optical flow input
        num_ftrs_temporal = self.temporal_stream.fc.in_features
        self.temporal_stream.fc = nn.Identity()  # Remove the final FC layer
        
        # Fusion and classification
        self.fusion = nn.Sequential(
            nn.Linear(num_ftrs_spatial + num_ftrs_temporal, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        # Activation for binary classification
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, spatial_input, temporal_input):
        # Process spatial stream
        spatial_features = self.spatial_stream(spatial_input)
        
        # Process temporal stream
        temporal_features = self.temporal_stream(temporal_input)
        
        # Concatenate features
        combined = torch.cat((spatial_features, temporal_features), dim=1)
        
        # Classification
        logits = self.fusion(combined)
        return self.sigmoid(logits)
    
    def get_features(self, spatial_input, temporal_input):
        """Extract features before final classification."""
        # Process spatial stream
        spatial_features = self.spatial_stream(spatial_input)
        
        # Process temporal stream
        temporal_features = self.temporal_stream(temporal_input)
        
        # Concatenate features
        combined = torch.cat((spatial_features, temporal_features), dim=1)
        return combined


class TimeSformer(nn.Module):
    """Simplified TimeSformer-like model for video deepfake detection."""
    
    def __init__(self, num_classes=1, num_frames=16, img_size=224, patch_size=16, dim=768):
        super(TimeSformer, self).__init__()
        
        # Parameters
        self.num_frames = num_frames
        self.img_size = img_size
        self.patch_size = patch_size
        self.dim = dim
        
        # Calculate number of patches
        num_patches = (img_size // patch_size) ** 2
        self.total_patches = num_patches * num_frames
        
        # Patch embedding
        self.patch_embed = nn.Conv3d(
            in_channels=3,
            out_channels=dim,
            kernel_size=(1, patch_size, patch_size),
            stride=(1, patch_size, patch_size),
            bias=False
        )
        
        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.total_patches + 1, dim))
        
        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        
        # Transformer layers - simplified for this implementation
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=8,
                dim_feedforward=dim * 4,
                dropout=0.1,
                activation='gelu'
            ),
            num_layers=6
        )
        
        # Classification head
        self.fc = nn.Linear(dim, num_classes)
        
        # Activation for binary classification
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x shape: [B, C, T, H, W]
        B, C, T, H, W = x.shape
        
        # Patch embedding
        x = self.patch_embed(x)  # [B, D, T, H//P, W//P]
        x = x.permute(0, 2, 3, 4, 1)  # [B, T, H//P, W//P, D]
        x = x.flatten(1, 3)  # [B, T*H//P*W//P, D]
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add position embedding
        x = x + self.pos_embed[:, :(x.size(1))]
        
        # Apply transformer
        x = x.permute(1, 0, 2)  # [L, B, D]
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # [B, L, D]
        
        # Classification
        x = x[:, 0]  # Take the class token output
        x = self.fc(x)
        
        return self.sigmoid(x)
    
    def get_features(self, x):
        """Extract features before final classification."""
        # x shape: [B, C, T, H, W]
        B, C, T, H, W = x.shape
        
        # Patch embedding
        x = self.patch_embed(x)  # [B, D, T, H//P, W//P]
        x = x.permute(0, 2, 3, 4, 1)  # [B, T, H//P, W//P, D]
        x = x.flatten(1, 3)  # [B, T*H//P*W//P, D]
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add position embedding
        x = x + self.pos_embed[:, :(x.size(1))]
        
        # Apply transformer
        x = x.permute(1, 0, 2)  # [L, B, D]
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # [B, L, D]
        
        # Take the class token output
        x = x[:, 0]
        
        return x


def get_video_model(model_type=None):
    """
    Get or initialize the video detection model.
    
    Args:
        model_type (str, optional): Type of model to use. If None, uses the default from settings.
    
    Returns:
        nn.Module: The loaded model
    """
    if model_type is None:
        model_type = settings.VIDEO_MODEL_TYPE
    
    # Check if model is already loaded
    if model_type in _video_models:
        return _video_models[model_type]
    
    # Create model based on type
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device} for video model")
    
    if model_type == "3dcnn":
        model = C3D()
        weights_path = os.path.join(settings.MODEL_WEIGHTS_DIR, "c3d_deepfake.pth")
    elif model_type == "two-stream":
        model = TwoStreamNetwork()
        weights_path = os.path.join(settings.MODEL_WEIGHTS_DIR, "two_stream_deepfake.pth")  # This file might not exist
    elif model_type == "timesformer":
        model = TimeSformer()
        weights_path = os.path.join(settings.MODEL_WEIGHTS_DIR, "timesformer_deepfake.pyth")  # Changed to .pyth
    else:
        raise ValueError(f"Unsupported video model type: {model_type}")
    
    # Load weights if available
    if os.path.exists(weights_path):
        logger.info(f"Loading weights from {weights_path}")
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        logger.warning(f"Weights file not found at {weights_path}, using model with default initialization")
    
    model = model.to(device)
    model.eval()
    
    # Cache the model
    _video_models[model_type] = model
    
    return model


def preprocess_video(video_path: str, model_type=None, max_frames=32) -> Dict[str, torch.Tensor]:
    """
    Preprocess video for deepfake detection.
    
    Args:
        video_path (str): Path to the video file
        model_type (str, optional): Type of model to use
        max_frames (int): Maximum number of frames to extract
    
    Returns:
        Dict[str, torch.Tensor]: Preprocessed video tensors for the model
    """
    if model_type is None:
        model_type = settings.VIDEO_MODEL_TYPE
    
    # Extract frames from video
    frames = extract_frames(video_path, max_frames=max_frames)
    
    # Extract faces from frames (optional, depends on your approach)
    # face_frames = extract_faces(frames)
    
    # For this simplified example, we'll use the full frames
    face_frames = frames
    
    # Convert to numpy arrays and normalize
    frame_tensor = np.array(face_frames) / 255.0
    
    # Rearrange to [num_frames, height, width, channels]
    frame_tensor = np.array(frame_tensor).astype(np.float32)
    
    if model_type == "3dcnn":
        # 3DCNN expects input shape [batch, channels, frames, height, width]
        frame_tensor = np.transpose(frame_tensor, (3, 0, 1, 2))  # [C, T, H, W]
        frame_tensor = np.expand_dims(frame_tensor, axis=0)  # [1, C, T, H, W]
        return {"video": torch.tensor(frame_tensor)}
    
    elif model_type == "two-stream":
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
            "spatial": torch.tensor(spatial_tensor),
            "temporal": torch.tensor(mean_flow)
        }
    
    elif model_type == "timesformer":
        # TimeSformer expects input shape [batch, channels, frames, height, width]
        frame_tensor = np.transpose(frame_tensor, (3, 0, 1, 2))  # [C, T, H, W]
        frame_tensor = np.expand_dims(frame_tensor, axis=0)  # [1, C, T, H, W]
        return {"video": torch.tensor(frame_tensor)}
    
    else:
        raise ValueError(f"Unsupported video model type: {model_type}")


def detect_deepfake_video(video_path: str, model_type=None, detailed=False) -> Dict[str, Any]:
    """
    Detect if a video is a deepfake.
    
    Args:
        video_path (str): Path to the video file
        model_type (str, optional): Type of model to use
        detailed (bool): Whether to return detailed analysis
    
    Returns:
        Dict: Detection results
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model_type is None:
        model_type = settings.VIDEO_MODEL_TYPE
    
    model = get_video_model(model_type)
    
    try:
        # Preprocess video
        video_tensors = preprocess_video(video_path, model_type)
        
        # Move tensors to device
        for key in video_tensors:
            video_tensors[key] = video_tensors[key].to(device)
        
        # Get prediction
        with torch.no_grad():
            if model_type == "two-stream":
                prediction = model(video_tensors["spatial"], video_tensors["temporal"])
            else:
                prediction = model(video_tensors["video"])
            
            confidence = prediction.item()
        
        # Determine if fake
        is_fake = confidence >= 0.5
        
        result = {
            "is_fake": is_fake,
            "confidence_score": confidence,
            "model_used": model_type,
        }
        
        # Add detailed analysis if requested
        if detailed:
            # Extract detailed information from video
            frames = extract_frames(video_path, max_frames=16)
            
            # Analyze individual frames
            frame_scores = []
            from app.models.image_models import get_image_model, preprocess_image
            
            image_model = get_image_model()
            
            for i, frame in enumerate(frames):
                # Save frame temporarily
                temp_frame_path = os.path.join(settings.UPLOAD_DIR, f"temp_frame_{i}.jpg")
                cv2.imwrite(temp_frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                
                # Process with image model
                try:
                    frame_tensor = preprocess_image(temp_frame_path).to(device)
                    with torch.no_grad():
                        frame_pred = image_model(frame_tensor)
                    frame_scores.append(frame_pred.item())
                except Exception as e:
                    logger.error(f"Error processing frame {i}: {str(e)}")
                    frame_scores.append(0.5)  # Default score
                
                # Remove temporary file
                os.remove(temp_frame_path)
            
            # Extract features from the video
            with torch.no_grad():
                if model_type == "two-stream":
                    features = model.get_features(video_tensors["spatial"], video_tensors["temporal"])
                else:
                    features = model.get_features(video_tensors["video"])
            
            # Record timestamp for each frame
            # Assuming 30 fps
            timestamps = [i * (1/30) for i in range(len(frame_scores))]
            
            result["detailed_analysis"] = {
                "feature_vector": features.detach().cpu().numpy()[0].tolist(),
                "temporal_analysis": {
                    "timestamps": timestamps,
                    "scores": frame_scores,
                    "threshold": 0.5
                }
            }
            
            # Calculate temporal inconsistency
            if len(frame_scores) > 1:
                inconsistency = np.std(frame_scores)
                result["detailed_analysis"]["temporal_inconsistency"]
                inconsistency = np.std(frame_scores)
                result["detailed_analysis"]["temporal_inconsistency"] = inconsistency
                result["detailed_analysis"]["temporal_consistency_score"] = 1.0 - (inconsistency * 2)  # Scale for easier interpretation
            
            # Check for lip sync inconsistencies if applicable
            if model_type == "3dcnn" or model_type == "timesformer":
                result["detailed_analysis"]["lip_sync_analysis"] = {
                    "lip_sync_score": 0.75,  # Placeholder for actual analysis
                    "mouth_movement_consistency": 0.82
                }
            
        return result
    
    except Exception as e:
        logger.error(f"Error during video deepfake detection: {str(e)}")
        return {
            "error": str(e),
            "is_fake": False,
            "confidence_score": 0.0
        }


def generate_video_visualization(video_path: str, model_type=None) -> Dict[str, Any]:
    """
    Generate visualizations for video deepfake detection.
    
    Args:
        video_path (str): Path to the video file
        model_type (str, optional): Type of model to use
    
    Returns:
        Dict: Visualization data including frame-by-frame analysis
    """
    # Extract frames
    frames = extract_frames(video_path, max_frames=16)
    
    # Analyze individual frames
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from app.models.image_models import get_image_model, preprocess_image
    
    image_model = get_image_model()
    
    frame_scores = []
    heatmaps = []
    
    for i, frame in enumerate(frames):
        # Save frame temporarily
        temp_frame_path = os.path.join(settings.UPLOAD_DIR, f"temp_frame_{i}.jpg")
        cv2.imwrite(temp_frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        
        # Process with image model
        try:
            frame_tensor = preprocess_image(temp_frame_path).to(device)
            with torch.no_grad():
                frame_pred = image_model(frame_tensor)
            frame_scores.append(frame_pred.item())
            
            # Generate heatmap for visualization
            from app.models.image_models import generate_heatmap
            _, heatmap = generate_heatmap(temp_frame_path)
            heatmaps.append(heatmap.tolist())
            
        except Exception as e:
            logger.error(f"Error processing frame {i}: {str(e)}")
            frame_scores.append(0.5)  # Default score
            heatmaps.append(np.zeros((frame.shape[0], frame.shape[1])).tolist())
        
        # Remove temporary file
        os.remove(temp_frame_path)
    
    # Timestamps (assuming 30 fps)
    timestamps = [i * (1/30) for i in range(len(frame_scores))]
    
    return {
        "frame_analysis": {
            "timestamps": timestamps,
            "scores": frame_scores,
            "threshold": 0.5
        },
        "heatmaps": heatmaps
    }