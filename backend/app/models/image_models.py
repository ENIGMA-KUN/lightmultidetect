import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from typing import Dict, Tuple, Any, List, Union, Optional
import logging

from app.core.config import settings

# Configure logging
logger = logging.getLogger(__name__)

# Define transforms for preprocessing
preprocess = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# Global model cache
_image_models = {}


class XceptionNet(nn.Module):
    """Xception-based deepfake detection model."""
    
    def __init__(self, num_classes=1):
        super(XceptionNet, self).__init__()
        # Load pretrained Xception model
        self.xception = models.xception(pretrained=True)
        
        # Replace final fully connected layer
        in_features = self.xception.fc.in_features
        self.xception.fc = nn.Linear(in_features, num_classes)
        
        # Add activation for binary classification
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.xception(x)
        return self.sigmoid(x)
    
    def get_features(self, x):
        """Extract features before final classification."""
        # Get the features just before the final fc layer
        x = self.xception.conv1(x)
        x = self.xception.bn1(x)
        x = self.xception.relu(x)
        
        x = self.xception.conv2(x)
        x = self.xception.bn2(x)
        x = self.xception.relu(x)
        
        x = self.xception.block1(x)
        x = self.xception.block2(x)
        x = self.xception.block3(x)
        x = self.xception.block4(x)
        x = self.xception.block5(x)
        x = self.xception.block6(x)
        x = self.xception.block7(x)
        x = self.xception.block8(x)
        x = self.xception.block9(x)
        x = self.xception.block10(x)
        x = self.xception.block11(x)
        x = self.xception.block12(x)
        
        x = self.xception.conv3(x)
        x = self.xception.bn3(x)
        x = self.xception.relu(x)
        
        x = self.xception.conv4(x)
        x = self.xception.bn4(x)
        x = self.xception.relu(x)
        
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        
        return x


class EfficientNetModel(nn.Module):
    """EfficientNet-based deepfake detection model."""
    
    def __init__(self, num_classes=1, variant='b4'):
        super(EfficientNetModel, self).__init__()
        # Load pretrained EfficientNet model
        if variant == 'b0':
            self.efficientnet = models.efficientnet_b0(pretrained=True)
        elif variant == 'b4':
            self.efficientnet = models.efficientnet_b4(pretrained=True)
        else:
            raise ValueError(f"Unsupported EfficientNet variant: {variant}")
        
        # Replace classifier
        in_features = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, num_classes)
        )
        
        # Add activation for binary classification
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.efficientnet(x)
        return self.sigmoid(x)
    
    def get_features(self, x):
        """Extract features before final classification."""
        # Get features from the EfficientNet backbone
        x = self.efficientnet.features(x)
        x = self.efficientnet.avgpool(x)
        x = torch.flatten(x, 1)
        return x


class MesoNet(nn.Module):
    """Lightweight MesoNet model for efficient deepfake detection."""
    
    def __init__(self, num_classes=1):
        super(MesoNet, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu = nn.ReLU(inplace=True)
        self.leakyrelu = nn.LeakyReLU(0.1)
        
        self.conv2 = nn.Conv2d(8, 8, 5, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(8)
        self.conv3 = nn.Conv2d(8, 16, 5, padding=2, bias=False)
        self.bn3 = nn.BatchNorm2d(16)
        self.conv4 = nn.Conv2d(16, 16, 5, padding=2, bias=False)
        self.bn4 = nn.BatchNorm2d(16)
        
        # Pooling layers
        self.maxpooling1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.maxpooling2 = nn.MaxPool2d(kernel_size=(4, 4))
        
        # Fully connected layers
        self.fc1 = nn.Linear(16 * 8 * 8, 16)
        self.dropout = nn.Dropout2d(0.5)
        self.fc2 = nn.Linear(16, num_classes)
        
        # Activation for binary classification
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(self.bn1(x))
        x = self.maxpooling1(x)
        
        x = self.conv2(x)
        x = self.relu(self.bn2(x))
        x = self.maxpooling1(x)
        
        x = self.conv3(x)
        x = self.relu(self.bn3(x))
        x = self.maxpooling1(x)
        
        x = self.conv4(x)
        x = self.relu(self.bn4(x))
        x = self.maxpooling2(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.leakyrelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return self.sigmoid(x)
    
    def get_features(self, x):
        """Extract features before final classification."""
        x = self.conv1(x)
        x = self.relu(self.bn1(x))
        x = self.maxpooling1(x)
        
        x = self.conv2(x)
        x = self.relu(self.bn2(x))
        x = self.maxpooling1(x)
        
        x = self.conv3(x)
        x = self.relu(self.bn3(x))
        x = self.maxpooling1(x)
        
        x = self.conv4(x)
        x = self.relu(self.bn4(x))
        x = self.maxpooling2(x)
        
        x = x.view(x.size(0), -1)
        return x


# Find this function
def get_image_model(model_type=None):
    """Get or initialize the image detection model."""
    if model_type is None:
        model_type = settings.IMAGE_MODEL_TYPE
    
    # Check if model is already loaded
    if model_type in _image_models:
        return _image_models[model_type]
    
    # Create model based on type
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device} for image model")
    
    if model_type == "xception":
        model = XceptionNet()
        weights_path = os.path.join(settings.MODEL_WEIGHTS_DIR, "xception_deepfake.pth")
    elif model_type == "efficientnet":
        model = EfficientNetModel(variant='b4')
        weights_path = os.path.join(settings.MODEL_WEIGHTS_DIR, "efficientnet_deepfake.h5")  # Changed to .h5
    elif model_type == "mesonet":
        model = MesoNet()
        weights_path = os.path.join(settings.MODEL_WEIGHTS_DIR, "mesonet_deepfake.h5")  # Changed to .h5
    else:
        raise ValueError(f"Unsupported image model type: {model_type}")
    
    # Load weights if available
    if os.path.exists(weights_path):
        logger.info(f"Loading weights from {weights_path}")
        if weights_path.endswith('.h5'):
            # For TensorFlow models, we need a different loading approach
            try:
                import tensorflow as tf
                import numpy as np
                
                # Load the h5 file
                tf_model = tf.keras.models.load_model(weights_path)
                
                # Extract weights and convert to PyTorch format
                # This is a simplified version - you may need to map specific layers
                for i, layer in enumerate(model.modules()):
                    if hasattr(layer, 'weight'):
                        # Get corresponding TF layer weights
                        if i < len(tf_model.layers):
                            tf_weights = tf_model.layers[i].get_weights()
                            if len(tf_weights) > 0:
                                # Convert and assign weights
                                layer.weight.data = torch.from_numpy(tf_weights[0].transpose())
                                if hasattr(layer, 'bias') and layer.bias is not None and len(tf_weights) > 1:
                                    layer.bias.data = torch.from_numpy(tf_weights[1])
            except ImportError:
                logger.error("TensorFlow not installed, cannot load .h5 weights")
                logger.warning("Using model with default initialization")
            except Exception as e:
                logger.error(f"Error loading .h5 weights: {str(e)}")
                logger.warning("Using model with default initialization")
        elif weights_path.endswith('.onnx'):
            try:
                import onnx
                import onnx2pytorch
                
                # Load ONNX model
                onnx_model = onnx.load(weights_path)
                
                # Convert to PyTorch
                pytorch_model = onnx2pytorch.ConvertModel(onnx_model)
                
                # Map the converted model to our model architecture
                # This requires careful mapping of layers
                model = pytorch_model
            except ImportError:
                logger.error("onnx or onnx2pytorch not installed, cannot load .onnx weights")
                logger.warning("Using model with default initialization")
            except Exception as e:
                logger.error(f"Error loading .onnx weights: {str(e)}")
                logger.warning("Using model with default initialization")
        else:
            # Standard PyTorch loading for .pth, .pt, and .pyth files
            try:
                state_dict = torch.load(weights_path, map_location=device)
                model.load_state_dict(state_dict)
            except Exception as e:
                logger.error(f"Error loading weights: {str(e)}")
                logger.warning("Using model with default initialization")
    else:
        logger.warning(f"Weights file not found at {weights_path}, using model with default initialization")
    
    model = model.to(device)
    model.eval()
    
    # Cache the model
    _image_models[model_type] = model
    
    return model


def preprocess_image(image_path: str) -> torch.Tensor:
    """
    Preprocess an image for the deepfake detection model.
    
    Args:
        image_path (str): Path to the image file
    
    Returns:
        torch.Tensor: Preprocessed image tensor
    """
    image = Image.open(image_path).convert('RGB')
    image_tensor = preprocess(image).unsqueeze(0)
    return image_tensor


def detect_deepfake(image_path: str, model_type=None, detailed=False) -> Dict[str, Any]:
    """
    Detect if an image is a deepfake.
    
    Args:
        image_path (str): Path to the image file
        model_type (str, optional): Type of model to use
        detailed (bool): Whether to return detailed analysis
    
    Returns:
        Dict: Detection results
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_image_model(model_type)
    
    try:
        # Preprocess image
        image_tensor = preprocess_image(image_path)
        image_tensor = image_tensor.to(device)
        
        # Get prediction
        with torch.no_grad():
            prediction = model(image_tensor)
            confidence = prediction.item()
        
        # Determine if fake
        is_fake = confidence >= 0.5
        
        result = {
            "is_fake": is_fake,
            "confidence_score": confidence,
            "model_used": model_type or settings.IMAGE_MODEL_TYPE,
        }
        
        # Add detailed analysis if requested
        if detailed:
            # Extract features
            with torch.no_grad():
                features = model.get_features(image_tensor)
            
            # Get gradients for visualization (Grad-CAM approach)
            image_tensor.requires_grad = True
            output = model(image_tensor)
            model.zero_grad()
            output.backward()
            gradients = image_tensor.grad.detach().cpu().numpy()[0]
            
            # Get average gradient values
            gradient_importance = np.mean(np.abs(gradients), axis=(1, 2))
            
            result["detailed_analysis"] = {
                "feature_vector": features.detach().cpu().numpy()[0].tolist(),
                "gradient_importance": gradient_importance.tolist(),
                "suspicious_regions": []  # Would be populated with region analysis
            }
            
            # Add more detailed analysis based on model type
            if model_type == "xception" or model_type is None:
                # Xception-specific details
                result["detailed_analysis"]["artifact_detection"] = {
                    "compression_artifacts": 0.35,  # Placeholder for real analysis
                    "noise_patterns": 0.42,
                    "texture_inconsistencies": 0.65
                }
            
        return result
    
    except Exception as e:
        logger.error(f"Error during deepfake detection: {str(e)}")
        return {
            "error": str(e),
            "is_fake": False,
            "confidence_score": 0.0
        }


def generate_heatmap(image_path: str, model_type=None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a heatmap highlighting suspected deepfake regions.
    
    Args:
        image_path (str): Path to the image file
        model_type (str, optional): Type of model to use
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: Original image and heatmap
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_image_model(model_type)
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)
    
    image_tensor = preprocess(image).unsqueeze(0)
    image_tensor = image_tensor.to(device)
    image_tensor.requires_grad = True
    
    # Forward pass
    output = model(image_tensor)
    
    # Backward pass for gradients
    model.zero_grad()
    output.backward()
    
    # Get gradients
    gradients = image_tensor.grad.detach().cpu().numpy()[0]
    
    # Get activations from the last convolutional layer
    # For a simple approximation, we'll use the gradient magnitude as the heatmap
    heatmap = np.mean(np.abs(gradients), axis=0)
    
    # Normalize heatmap
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    heatmap = np.uint8(255 * heatmap)
    
    # Resize heatmap to match original image
    heatmap = Image.fromarray(heatmap).resize((image_np.shape[1], image_np.shape[0]))
    heatmap_np = np.array(heatmap)
    
    return image_np, heatmap_np