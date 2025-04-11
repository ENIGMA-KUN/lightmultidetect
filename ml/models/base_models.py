# ml/models/base_models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from efficientnet_pytorch import EfficientNet


class LightweightImageEncoder(nn.Module):
    """
    Lightweight image encoder using MobileNetV3 with frequency domain attention
    """
    def __init__(self, pretrained=True, feature_dim=512):
        super().__init__()
        # Use MobileNetV3 Small for efficiency
        self.backbone = models.mobilenet_v3_small(pretrained=pretrained)
        
        # Replace classifier with feature extractor
        self.backbone.classifier = nn.Sequential(
            nn.Linear(576, feature_dim),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2, inplace=True),
        )
        
        # Frequency domain attention module
        self.freq_attention = FrequencyDomainAttention(feature_dim)
        
    def forward(self, x):
        # Extract features using backbone
        features = self.backbone(x)
        
        # Apply frequency domain attention
        enhanced_features = self.freq_attention(features)
        
        return enhanced_features


class FrequencyDomainAttention(nn.Module):
    """
    Attention module that focuses on frequency domain artifacts, which are common in deepfakes
    """
    def __init__(self, feature_dim):
        super().__init__()
        
        # Frequency attention mechanism
        self.frequency_gate = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // 4, feature_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Generate attention weights
        attention_weights = self.frequency_gate(x)
        
        # Apply attention
        enhanced_features = x * attention_weights
        
        return enhanced_features


class LightweightAudioEncoder(nn.Module):
    """
    Lightweight audio encoder for processing mel-spectrograms
    """
    def __init__(self, input_channels=1, feature_dim=512):
        super().__init__()
        
        # Convolutional layers for spectrogram processing
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        
        # Adaptive pooling to handle variable input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Feature projection
        self.fc = nn.Sequential(
            nn.Linear(128 * 4 * 4, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
    def forward(self, x):
        # Apply convolution blocks
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        # Apply adaptive pooling
        x = self.adaptive_pool(x)
        
        # Flatten and project
        x = x.view(x.size(0), -1)
        features = self.fc(x)
        
        return features


class LightweightVideoEncoder(nn.Module):
    """
    Lightweight video encoder with temporal modeling
    """
    def __init__(self, pretrained=True, feature_dim=512, num_frames=16):
        super().__init__()
        
        # Image feature extractor (frame-level)
        self.image_encoder = LightweightImageEncoder(pretrained=pretrained, feature_dim=feature_dim)
        
        # Temporal modeling
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        # Temporal attention
        self.temporal_attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # Frame count normalization
        self.num_frames = num_frames
        
    def forward(self, x):
        batch_size, num_frames, channels, height, width = x.shape
        
        # Process each frame
        frame_features = []
        for i in range(num_frames):
            frame = x[:, i, :, :, :]
            frame_feature = self.image_encoder(frame)
            frame_features.append(frame_feature)
            
        # Stack frame features
        frame_features = torch.stack(frame_features, dim=2)  # [batch_size, feature_dim, num_frames]
        
        # Apply temporal convolution
        temporal_features = self.temporal_conv(frame_features)
        
        # Calculate temporal attention
        reshaped_features = temporal_features.permute(0, 2, 1)  # [batch_size, num_frames, feature_dim]
        attention_weights = []
        
        for i in range(num_frames):
            weight = self.temporal_attention(reshaped_features[:, i, :])
            attention_weights.append(weight)
            
        attention_weights = torch.cat(attention_weights, dim=1)  # [batch_size, num_frames]
        attention_weights = F.softmax(attention_weights, dim=1).unsqueeze(1)  # [batch_size, 1, num_frames]
        
        # Apply attention weights
        weighted_features = torch.bmm(attention_weights, reshaped_features)  # [batch_size, 1, feature_dim]
        video_features = weighted_features.squeeze(1)  # [batch_size, feature_dim]
        
        return video_features


class MultiModalFusionNetwork(nn.Module):
    """
    Multi-modal fusion network combining image, audio, and video features
    """
    def __init__(self, feature_dim=512, num_classes=2):
        super().__init__()
        
        # Modal-specific encoders
        self.image_encoder = LightweightImageEncoder(feature_dim=feature_dim)
        self.audio_encoder = LightweightAudioEncoder(feature_dim=feature_dim)
        self.video_encoder = LightweightVideoEncoder(feature_dim=feature_dim)
        
        # Cross-modal attention
        self.cross_modal_attention = CrossModalAttention(feature_dim)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim * 3, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(feature_dim, num_classes)
        )
        
        # Modal weighting module (for dynamic weighting of modalities)
        self.modal_weights = nn.Sequential(
            nn.Linear(feature_dim * 3, 3),
            nn.Softmax(dim=1)
        )
        
    def forward(self, image=None, audio=None, video=None):
        # Extract features from available modalities
        image_features = self.image_encoder(image) if image is not None else torch.zeros(video.size(0), 512).to(video.device)
        audio_features = self.audio_encoder(audio) if audio is not None else torch.zeros(video.size(0), 512).to(video.device)
        video_features = self.video_encoder(video) if video is not None else torch.zeros(image.size(0), 512).to(image.device)
        
        # Combine features
        combined_features = torch.cat([image_features, audio_features, video_features], dim=1)
        
        # Calculate dynamic modal weights
        weights = self.modal_weights(combined_features)
        
        # Apply cross-modal attention
        enhanced_features = self.cross_modal_attention(image_features, audio_features, video_features, weights)
        
        # Final classification
        output = self.fusion(enhanced_features)
        
        return output, weights


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention mechanism for integrating information across modalities
    """
    def __init__(self, feature_dim):
        super().__init__()
        
        # Query, key, value projections
        self.query_proj = nn.Linear(feature_dim, feature_dim)
        self.key_proj = nn.Linear(feature_dim, feature_dim)
        self.value_proj = nn.Linear(feature_dim, feature_dim)
        
        # Output projection
        self.output_proj = nn.Linear(feature_dim * 3, feature_dim * 3)
        
    def forward(self, image_features, audio_features, video_features, weights):
        # Project features
        image_query = self.query_proj(image_features)
        audio_query = self.query_proj(audio_features)
        video_query = self.query_proj(video_features)
        
        image_key = self.key_proj(image_features)
        audio_key = self.key_proj(audio_features)
        video_key = self.key_proj(video_features)
        
        image_value = self.value_proj(image_features)
        audio_value = self.value_proj(audio_features)
        video_value = self.value_proj(video_features)
        
        # Calculate attention scores
        i2a_attn = torch.bmm(image_query.unsqueeze(1), audio_key.unsqueeze(2)).squeeze(-1)
        i2v_attn = torch.bmm(image_query.unsqueeze(1), video_key.unsqueeze(2)).squeeze(-1)
        
        a2i_attn = torch.bmm(audio_query.unsqueeze(1), image_key.unsqueeze(2)).squeeze(-1)
        a2v_attn = torch.bmm(audio_query.unsqueeze(1), video_key.unsqueeze(2)).squeeze(-1)
        
        v2i_attn = torch.bmm(video_query.unsqueeze(1), image_key.unsqueeze(2)).squeeze(-1)
        v2a_attn = torch.bmm(video_query.unsqueeze(1), audio_key.unsqueeze(2)).squeeze(-1)
        
        # Apply softmax
        i2a_attn = F.softmax(i2a_attn, dim=1)
        i2v_attn = F.softmax(i2v_attn, dim=1)
        a2i_attn = F.softmax(a2i_attn, dim=1)
        a2v_attn = F.softmax(a2v_attn, dim=1)
        v2i_attn = F.softmax(v2i_attn, dim=1)
        v2a_attn = F.softmax(v2a_attn, dim=1)
        
        # Apply attention
        image_features_enhanced = image_features + i2a_attn * audio_value + i2v_attn * video_value
        audio_features_enhanced = audio_features + a2i_attn * image_value + a2v_attn * video_value
        video_features_enhanced = video_features + v2i_attn * image_value + v2a_attn * audio_value
        
        # Concatenate enhanced features
        enhanced_features = torch.cat([
            image_features_enhanced * weights[:, 0].unsqueeze(1),
            audio_features_enhanced * weights[:, 1].unsqueeze(1),
            video_features_enhanced * weights[:, 2].unsqueeze(1)
        ], dim=1)
        
        # Apply output projection
        enhanced_features = self.output_proj(enhanced_features)
        
        return enhanced_features


class LightMultiDetect(nn.Module):
    """
    Unified lightweight model for deepfake detection across modalities
    """
    def __init__(self, num_classes=2):
        super().__init__()
        
        self.fusion_network = MultiModalFusionNetwork(num_classes=num_classes)
        
    def forward(self, image=None, audio=None, video=None):
        return self.fusion_network(image, audio, video)


# Teacher-Student Knowledge Distillation

class TeacherModel(nn.Module):
    """
    Teacher model based on Xception for knowledge distillation
    """
    def __init__(self, num_classes=2):
        super().__init__()
        
        # Base model - Xception
        self.base_model = models.xception(pretrained=True)
        
        # Replace final classifier
        self.base_model.fc = nn.Linear(2048, num_classes)
        
    def forward(self, x):
        return self.base_model(x)


def distillation_loss(student_logits, teacher_logits, targets, alpha=0.5, temperature=2.0):
    """
    Knowledge distillation loss combining soft and hard targets
    
    Args:
        student_logits: Logits from the student model
        teacher_logits: Logits from the teacher model
        targets: Ground truth labels
        alpha: Weight for hard/soft loss components
        temperature: Temperature for softening probability distributions
        
    Returns:
        Combined distillation loss
    """
    # Hard loss (standard cross-entropy with true labels)
    hard_loss = F.cross_entropy(student_logits, targets)
    
    # Soft loss (KL divergence with teacher outputs)
    soft_targets = F.softmax(teacher_logits / temperature, dim=1)
    soft_prob = F.log_softmax(student_logits / temperature, dim=1)
    soft_loss = F.kl_div(soft_prob, soft_targets, reduction='batchmean') * (temperature ** 2)
    
    # Combined loss
    loss = alpha * hard_loss + (1 - alpha) * soft_loss
    
    return loss