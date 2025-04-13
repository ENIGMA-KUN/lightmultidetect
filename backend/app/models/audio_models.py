import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
import logging
from typing import Dict, Tuple, Any, List, Union, Optional

from app.core.config import settings

# Configure logging
logger = logging.getLogger(__name__)

# Global model cache
_audio_models = {}


class Wav2Vec2Model(nn.Module):
    """Wav2Vec 2.0 based deepfake audio detection model."""
    
    def __init__(self, num_classes=1):
        super(Wav2Vec2Model, self).__init__()
        # Initialize a wav2vec 2.0 model
        try:
            from transformers import Wav2Vec2Model as HFWav2Vec2Model
            self.wav2vec = HFWav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
            
            # Freeze most of the model
            for param in self.wav2vec.parameters():
                param.requires_grad = False
                
            # Unfreeze the last few layers
            for param in self.wav2vec.encoder.layers[-2:].parameters():
                param.requires_grad = True
            
            # Add classification head
            self.classifier = nn.Sequential(
                nn.Linear(768, 256),  # 768 is the hidden size of wav2vec2-base
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(64, num_classes)
            )
            
            # Add activation for binary classification
            self.sigmoid = nn.Sigmoid()
            
        except ImportError:
            logger.error("transformers package not found. Please install it with: pip install transformers")
            raise
    
    def forward(self, x):
        # wav2vec2 expects raw waveform input
        outputs = self.wav2vec(x)
        hidden_states = outputs.last_hidden_state
        
        # Pool over time dimension
        pooled = torch.mean(hidden_states, dim=1)
        
        # Apply classifier
        logits = self.classifier(pooled)
        return self.sigmoid(logits)
    
    def get_features(self, x):
        """Extract features before final classification."""
        outputs = self.wav2vec(x)
        hidden_states = outputs.last_hidden_state
        pooled = torch.mean(hidden_states, dim=1)
        return pooled


class RawNet2(nn.Module):
    """RawNet2 model for raw waveform analysis."""
    
    def __init__(self, num_classes=1):
        super(RawNet2, self).__init__()
        
        # First convolutional block
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.3)
        )
        
        # Residual blocks
        self.res_block1 = self._make_res_block(32, 32)
        self.res_block2 = self._make_res_block(32, 64, stride=2)
        self.res_block3 = self._make_res_block(64, 64)
        self.res_block4 = self._make_res_block(64, 128, stride=2)
        self.res_block5 = self._make_res_block(128, 128)
        self.res_block6 = self._make_res_block(128, 256, stride=2)
        
        # Global pooling and classification
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.sigmoid = nn.Sigmoid()
    
    def _make_res_block(self, in_channels, out_channels, stride=1):
        layers = []
        layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1))
        layers.append(nn.BatchNorm1d(out_channels))
        layers.append(nn.LeakyReLU(0.3))
        layers.append(nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
        layers.append(nn.BatchNorm1d(out_channels))
        
        # Skip connection
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.skip = None
        
        layers.append(nn.LeakyReLU(0.3))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Ensure input is [B, 1, T] where B is batch size and T is time samples
        x = self.conv1(x)
        
        # Apply residual blocks
        if self.skip is not None:
            x = self.res_block1(x) + self.skip(x)
        else:
            x = self.res_block1(x) + x
        
        self.skip = None
        x = self.res_block2(x)
        
        self.skip = None
        x = self.res_block3(x) + x
        
        self.skip = None
        x = self.res_block4(x)
        
        self.skip = None
        x = self.res_block5(x) + x
        
        self.skip = None
        x = self.res_block6(x)
        
        # Global pooling
        x = self.global_pool(x).squeeze(-1)
        
        # Classification
        x = F.leaky_relu(self.fc1(x), 0.3)
        x = self.fc2(x)
        
        return self.sigmoid(x)
    
    def get_features(self, x):
        # Ensure input is [B, 1, T] where B is batch size and T is time samples
        x = self.conv1(x)
        
        # Apply residual blocks
        if self.skip is not None:
            x = self.res_block1(x) + self.skip(x)
        else:
            x = self.res_block1(x) + x
        
        self.skip = None
        x = self.res_block2(x)
        
        self.skip = None
        x = self.res_block3(x) + x
        
        self.skip = None
        x = self.res_block4(x)
        
        self.skip = None
        x = self.res_block5(x) + x
        
        self.skip = None
        x = self.res_block6(x)
        
        # Global pooling
        x = self.global_pool(x).squeeze(-1)
        
        return x


class MelSpecResNet(nn.Module):
    """ResNet-based model for mel-spectrogram analysis."""
    
    def __init__(self, num_classes=1):
        super(MelSpecResNet, self).__init__()
        # Load a pretrained ResNet-18
        import torchvision.models as models
        self.resnet = models.resnet18(pretrained=True)
        
        # Modify first convolutional layer to accept single-channel input
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Modify final fully connected layer
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)
        
        # Add activation for binary classification
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # ResNet expects input of shape [B, C, H, W]
        x = self.resnet(x)
        return self.sigmoid(x)
    
    def get_features(self, x):
        """Extract features before final classification."""
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        
        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        
        return x


def get_audio_model(model_type=None):
    """
    Get or initialize the audio detection model.
    
    Args:
        model_type (str, optional): Type of model to use. If None, uses the default from settings.
    
    Returns:
        nn.Module: The loaded model
    """
    if model_type is None:
        model_type = settings.AUDIO_MODEL_TYPE
    
    # Check if model is already loaded
    if model_type in _audio_models:
        return _audio_models[model_type]
    
    # Create model based on type
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device} for audio model")
    
    if model_type == "wav2vec2":
        model = Wav2Vec2Model()
        weights_path = os.path.join(settings.MODEL_WEIGHTS_DIR, "wav2vec2_deepfake.pt")  # Changed to .pt
    elif model_type == "rawnet2":
        model = RawNet2()
        weights_path = os.path.join(settings.MODEL_WEIGHTS_DIR, "rawnet2_deepfake.pth") 
    elif model_type == "melspec":
        model = MelSpecResNet()
        weights_path = os.path.join(settings.MODEL_WEIGHTS_DIR, "melspec_deepfake.onnx")  # Changed to .onnx
    else:
        raise ValueError(f"Unsupported audio model type: {model_type}")
    
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
    _audio_models[model_type] = model
    
    return model


def preprocess_audio(audio_path: str, model_type=None) -> Union[torch.Tensor, Tuple[torch.Tensor, int]]:
    """
    Preprocess audio for the deepfake detection model.
    
    Args:
        audio_path (str): Path to the audio file
        model_type (str, optional): Type of model to use
    
    Returns:
        Union[torch.Tensor, Tuple[torch.Tensor, int]]: Preprocessed audio tensor and sample rate if needed
    """
    if model_type is None:
        model_type = settings.AUDIO_MODEL_TYPE
    
    # Load audio file
    y, sr = librosa.load(audio_path, sr=16000)  # Resample to 16kHz
    
    if model_type == "wav2vec2":
        # Wav2Vec2 expects raw waveform input
        if len(y) > 16000 * 10:  # Limit to 10 seconds
            y = y[:16000 * 10]
        audio_tensor = torch.tensor(y).unsqueeze(0)  # Add batch dimension
        return audio_tensor, sr
    
    elif model_type == "rawnet2":
        # RawNet2 expects raw waveform input with shape [B, 1, T]
        if len(y) > 16000 * 10:  # Limit to 10 seconds
            y = y[:16000 * 10]
        audio_tensor = torch.tensor(y).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        return audio_tensor
    
    elif model_type == "melspec":
        # Convert to mel spectrogram
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        S_db = librosa.power_to_db(S, ref=np.max)
        
        # Normalize
        S_db = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-8)
        
        # Convert to tensor with shape [B, C, H, W]
        audio_tensor = torch.tensor(S_db).unsqueeze(0).unsqueeze(0).float()
        return audio_tensor
    
    else:
        raise ValueError(f"Unsupported audio model type: {model_type}")


def detect_deepfake_audio(audio_path: str, model_type=None, detailed=False) -> Dict[str, Any]:
    """
    Detect if an audio file is a deepfake.
    
    Args:
        audio_path (str): Path to the audio file
        model_type (str, optional): Type of model to use
        detailed (bool): Whether to return detailed analysis
    
    Returns:
        Dict: Detection results
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model_type is None:
        model_type = settings.AUDIO_MODEL_TYPE
    
    model = get_audio_model(model_type)
    
    try:
        # Preprocess audio
        if model_type == "wav2vec2":
            audio_tensor, sr = preprocess_audio(audio_path, model_type)
            audio_tensor = audio_tensor.to(device)
        else:
            audio_tensor = preprocess_audio(audio_path, model_type)
            audio_tensor = audio_tensor.to(device)
        
        # Get prediction
        with torch.no_grad():
            prediction = model(audio_tensor)
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
            # Extract features
            with torch.no_grad():
                features = model.get_features(audio_tensor)
            
            # Load audio for temporal analysis
            y, sr = librosa.load(audio_path, sr=16000)
            
            # Calculate additional features
            spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr).mean(axis=1).tolist()
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean(axis=1).tolist()
            
            # Time segments analysis
            segment_length = sr * 1  # 1 second segments
            num_segments = min(10, len(y) // segment_length)
            segment_scores = []
            
            for i in range(num_segments):
                segment = y[i * segment_length:(i + 1) * segment_length]
                segment_tensor = torch.tensor(segment).unsqueeze(0)
                
                if model_type == "wav2vec2":
                    segment_tensor = segment_tensor.to(device)
                    with torch.no_grad():
                        segment_pred = model(segment_tensor)
                elif model_type == "rawnet2":
                    segment_tensor = segment_tensor.unsqueeze(0).to(device)  # Add channel dimension
                    with torch.no_grad():
                        segment_pred = model(segment_tensor)
                else:
                    # For spectrogram models, we need to convert segment to spectrogram
                    S = librosa.feature.melspectrogram(y=segment, sr=sr, n_mels=128, fmax=8000)
                    S_db = librosa.power_to_db(S, ref=np.max)
                    S_db = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-8)
                    segment_tensor = torch.tensor(S_db).unsqueeze(0).unsqueeze(0).float().to(device)
                    with torch.no_grad():
                        segment_pred = model(segment_tensor)
                
                segment_scores.append(segment_pred.item())
            
            # Record timestamp for each segment
            timestamps = [i * segment_length / sr for i in range(num_segments)]
            
            result["detailed_analysis"] = {
                "feature_vector": features.detach().cpu().numpy()[0].tolist(),
                "spectral_contrast": spectral_contrast,
                "mfccs": mfccs,
                "temporal_analysis": {
                    "timestamps": timestamps,
                    "scores": segment_scores,
                    "threshold": 0.5
                }
            }
            
            # Add model-specific details
            if model_type == "wav2vec2":
                result["detailed_analysis"]["voice_characteristics"] = {
                    "pitch_consistency": 0.78,  # Placeholder values
                    "formant_analysis": 0.65,
                    "speech_rhythm": 0.82
                }
            
        return result
    
    except Exception as e:
        logger.error(f"Error during audio deepfake detection: {str(e)}")
        return {
            "error": str(e),
            "is_fake": False,
            "confidence_score": 0.0
        }


def generate_audio_visualization(audio_path: str, model_type=None) -> Dict[str, Any]:
    """
    Generate visualizations for audio deepfake detection.
    
    Args:
        audio_path (str): Path to the audio file
        model_type (str, optional): Type of model to use
    
    Returns:
        Dict: Visualization data including spectrogram and confidence over time
    """
    # Load audio
    y, sr = librosa.load(audio_path, sr=16000)
    
    # Generate mel spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    S_db = librosa.power_to_db(S, ref=np.max)
    
    # Normalize
    S_db_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-8)
    
    # Convert to numpy array for visualization
    spectrogram = S_db_norm
    
    # Analyze audio in segments
    segment_length = sr * 1  # 1 second segments
    num_segments = min(20, len(y) // segment_length)
    
    # Get model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_audio_model(model_type)
    
    # Process each segment
    timestamps = []
    confidence_scores = []
    
    for i in range(num_segments):
        segment = y[i * segment_length:(i + 1) * segment_length]
        segment_tensor = preprocess_audio(audio_path, model_type)  # This needs to be adapted
        segment_tensor = segment_tensor.to(device)
        
        with torch.no_grad():
            prediction = model(segment_tensor)
            confidence = prediction.item()
        
        timestamps.append(i)
        confidence_scores.append(confidence)
    
    return {
        "spectrogram": spectrogram.tolist(),
        "temporal_analysis": {
            "timestamps": timestamps,
            "confidence_scores": confidence_scores,
            "threshold": 0.5
        }
    }