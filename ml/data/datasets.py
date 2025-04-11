# ml/data/datasets.py
import os
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import librosa
import cv2
from tqdm import tqdm
import io

from ml.preprocessing.image_preprocessing import ImagePreprocessor
from ml.preprocessing.audio_preprocessing import AudioPreprocessor
from ml.preprocessing.video_preprocessing import VideoPreprocessor


class DeepfakeDataset(Dataset):
    """
    Dataset for image-based deepfake detection
    """
    def __init__(self, root_dir, split='train', transform=True, image_size=224):
        """
        Initialize dataset
        
        Args:
            root_dir: Root directory containing dataset
            split: Data split ('train', 'val', or 'test')
            transform: Whether to apply data transformations
            image_size: Size of input images
        """
        self.root_dir = root_dir
        self.split = split
        self.image_size = image_size
        
        # Define transforms
        if transform:
            if split == 'train':
                self.transform = transforms.Compose([
                    transforms.Resize((image_size, image_size)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomApply([
                        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)
                    ], p=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    transforms.RandomApply([
                        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.0))
                    ], p=0.2),
                    transforms.RandomApply([
                        AddGaussianNoise(mean=0.0, std=0.01)
                    ], p=0.2),
                    transforms.RandomApply([
                        AddJPEGNoise(quality_lower=70, quality_upper=90)
                    ], p=0.2)
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
        else:
            self.transform = None
        
        # Load data paths
        self.data = self._load_dataset()
        
    def _load_dataset(self):
        """
        Load dataset paths and labels
        
        Returns:
            List of (path, label) pairs
        """
        # Load from JSON if available (faster)
        json_path = os.path.join(self.root_dir, f'{self.split}_data.json')
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                data = json.load(f)
                return data
        
        # Otherwise manually collect files
        data = []
        
        # Process real images
        real_dir = os.path.join(self.root_dir, 'real')
        if os.path.exists(real_dir):
            for img_file in tqdm(os.listdir(real_dir), desc=f'Loading real {self.split} data'):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    data.append({
                        'path': os.path.join(real_dir, img_file),
                        'label': 0  # 0 for real
                    })
        
        # Process fake images
        fake_dir = os.path.join(self.root_dir, 'fake')
        if os.path.exists(fake_dir):
            for img_file in tqdm(os.listdir(fake_dir), desc=f'Loading fake {self.split} data'):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    data.append({
                        'path': os.path.join(fake_dir, img_file),
                        'label': 1  # 1 for fake
                    })
        
        # Check for specific dataset structure (FaceForensics++ style)
        ff_real_dir = os.path.join(self.root_dir, 'original', 'frames')
        ff_fake_methods = ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']
        
        if os.path.exists(ff_real_dir):
            for video_id in tqdm(os.listdir(ff_real_dir), desc=f'Loading FF++ real {self.split} data'):
                video_frames_dir = os.path.join(ff_real_dir, video_id)
                if os.path.isdir(video_frames_dir):
                    frame_files = [f for f in os.listdir(video_frames_dir) if f.endswith('.png')]
                    
                    # If too many frames, sample a subset
                    if len(frame_files) > 30:
                        if self.split == 'train':
                            # For training, sample random frames
                            frame_files = random.sample(frame_files, 30)
                        else:
                            # For val/test, sample evenly
                            frame_files = frame_files[::len(frame_files)//30][:30]
                    
                    for frame_file in frame_files:
                        data.append({
                            'path': os.path.join(video_frames_dir, frame_file),
                            'label': 0,  # 0 for real
                            'video_id': video_id,
                            'dataset': 'FaceForensics++'
                        })
        
        # Process fake methods
        for method in ff_fake_methods:
            method_dir = os.path.join(self.root_dir, method, 'frames')
            if os.path.exists(method_dir):
                for video_id in tqdm(os.listdir(method_dir), desc=f'Loading FF++ {method} {self.split} data'):
                    video_frames_dir = os.path.join(method_dir, video_id)
                    if os.path.isdir(video_frames_dir):
                        frame_files = [f for f in os.listdir(video_frames_dir) if f.endswith('.png')]
                        
                        # If too many frames, sample a subset
                        if len(frame_files) > 30:
                            if self.split == 'train':
                                # For training, sample random frames
                                frame_files = random.sample(frame_files, 30)
                            else:
                                # For val/test, sample evenly
                                frame_files = frame_files[::len(frame_files)//30][:30]
                        
                        for frame_file in frame_files:
                            data.append({
                                'path': os.path.join(video_frames_dir, frame_file),
                                'label': 1,  # 1 for fake
                                'video_id': video_id,
                                'method': method,
                                'dataset': 'FaceForensics++'
                            })
        
        # Check for Celeb-DF style structure
        celeb_real_dir = os.path.join(self.root_dir, 'Celeb-real', 'frames')
        celeb_fake_dir = os.path.join(self.root_dir, 'Celeb-synthesis', 'frames')
        
        if os.path.exists(celeb_real_dir):
            for video_id in tqdm(os.listdir(celeb_real_dir), desc=f'Loading Celeb-DF real {self.split} data'):
                video_frames_dir = os.path.join(celeb_real_dir, video_id)
                if os.path.isdir(video_frames_dir):
                    frame_files = [f for f in os.listdir(video_frames_dir) if f.endswith('.png') or f.endswith('.jpg')]
                    
                    # If too many frames, sample a subset
                    if len(frame_files) > 30:
                        if self.split == 'train':
                            # For training, sample random frames
                            frame_files = random.sample(frame_files, 30)
                        else:
                            # For val/test, sample evenly
                            frame_files = frame_files[::len(frame_files)//30][:30]
                    
                    for frame_file in frame_files:
                        data.append({
                            'path': os.path.join(video_frames_dir, frame_file),
                            'label': 0,  # 0 for real
                            'video_id': video_id,
                            'dataset': 'Celeb-DF'
                        })
        
        if os.path.exists(celeb_fake_dir):
            for video_id in tqdm(os.listdir(celeb_fake_dir), desc=f'Loading Celeb-DF fake {self.split} data'):
                video_frames_dir = os.path.join(celeb_fake_dir, video_id)
                if os.path.isdir(video_frames_dir):
                    frame_files = [f for f in os.listdir(video_frames_dir) if f.endswith('.png') or f.endswith('.jpg')]
                    
                    # If too many frames, sample a subset
                    if len(frame_files) > 30:
                        if self.split == 'train':
                            # For training, sample random frames
                            frame_files = random.sample(frame_files, 30)
                        else:
                            # For val/test, sample evenly
                            frame_files = frame_files[::len(frame_files)//30][:30]
                    
                    for frame_file in frame_files:
                        data.append({
                            'path': os.path.join(video_frames_dir, frame_file),
                            'label': 1,  # 1 for fake
                            'video_id': video_id,
                            'dataset': 'Celeb-DF'
                        })
        
        # Save to JSON for faster loading next time
        with open(json_path, 'w') as f:
            json.dump(data, f)
            
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Get dataset item
        
        Args:
            idx: Index
            
        Returns:
            Tuple of (image, label)
        """
        item = self.data[idx]
        image_path = item['path']
        label = item['label']
        
        # Load image
        try:
            image = Image.open(image_path).convert('RGB')
            
            # Apply transforms
            if self.transform:
                image = self.transform(image)
                
            return image, label
            
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a default item
            image = torch.zeros(3, self.image_size, self.image_size)
            return image, label


class MultiModalDeepfakeDataset(Dataset):
    """
    Dataset for multi-modal deepfake detection
    """
    def __init__(self, root_dir, split='train', transform=True, image_size=224, num_frames=16, max_audio_len=10,
                 sample_rate=16000, modalities=None):
        """
        Initialize dataset
        
        Args:
            root_dir: Root directory containing dataset
            split: Data split ('train', 'val', or 'test')
            transform: Whether to apply data transformations
            image_size: Size of input images
            num_frames: Number of frames to extract from videos
            max_audio_len: Maximum audio length in seconds
            sample_rate: Audio sample rate
            modalities: List of modalities to use ('image', 'audio', 'video')
        """
        self.root_dir = root_dir
        self.split = split
        self.image_size = image_size
        self.num_frames = num_frames
        self.max_audio_len = max_audio_len
        self.sample_rate = sample_rate
        
        # Set default modalities if not specified
        if modalities is None:
            self.modalities = ['image', 'audio', 'video']
        else:
            self.modalities = modalities
        
        # Initialize preprocessors
        self.image_processor = ImagePreprocessor(face_size=image_size)
        self.audio_processor = AudioPreprocessor(sample_rate=sample_rate)
        self.video_processor = VideoPreprocessor(face_size=image_size, num_frames=num_frames, sample_rate=sample_rate)
        
        # Define image transforms
        if transform:
            if split == 'train':
                self.transform = transforms.Compose([
                    transforms.Resize((image_size, image_size)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomApply([
                        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)
                    ], p=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    transforms.RandomApply([
                        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.0))
                    ], p=0.2),
                    transforms.RandomApply([
                        AddGaussianNoise(mean=0.0, std=0.01)
                    ], p=0.2),
                    transforms.RandomApply([
                        AddJPEGNoise(quality_lower=70, quality_upper=90)
                    ], p=0.2)
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
        else:
            self.transform = None
        
        # Load data paths
        self.data = self._load_dataset()
        
    def _load_dataset(self):
        """
        Load dataset paths and labels
        
        Returns:
            List of items with paths and labels
        """
        # Load from JSON if available
        json_path = os.path.join(self.root_dir, f'{self.split}_multimodal_data.json')
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                data = json.load(f)
                return data
        
        # Otherwise build dataset
        data = []
        
        # Process directories for different datasets
        dataset_patterns = [
            # FaceForensics++ pattern
            {
                'real_pattern': os.path.join(self.root_dir, 'original', 'videos'),
                'fake_patterns': [
                    os.path.join(self.root_dir, method, 'videos') 
                    for method in ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']
                ],
                'dataset_name': 'FaceForensics++'
            },
            # Celeb-DF pattern
            {
                'real_pattern': os.path.join(self.root_dir, 'Celeb-real', 'videos'),
                'fake_patterns': [os.path.join(self.root_dir, 'Celeb-synthesis', 'videos')],
                'dataset_name': 'Celeb-DF'
            },
            # DFDC pattern
            {
                'real_pattern': os.path.join(self.root_dir, 'DFDC', 'real', 'videos'),
                'fake_patterns': [os.path.join(self.root_dir, 'DFDC', 'fake', 'videos')],
                'dataset_name': 'DFDC'
            }
        ]
        
        # Process each dataset pattern
        for pattern in dataset_patterns:
            # Process real videos
            real_dir = pattern['real_pattern']
            if os.path.exists(real_dir):
                for vid_file in tqdm(os.listdir(real_dir), desc=f"Loading {pattern['dataset_name']} real videos"):
                    if vid_file.lower().endswith(('.mp4', '.avi', '.mov')):
                        # Extract image, audio, and video paths
                        video_path = os.path.join(real_dir, vid_file)
                        
                        # Derive frame path (if frames are extracted separately)
                        frames_dir = real_dir.replace('videos', 'frames')
                        frame_path = os.path.join(frames_dir, os.path.splitext(vid_file)[0])
                        
                        # Get a sample frame if possible
                        sample_frame = None
                        if os.path.exists(frame_path):
                            frame_files = [f for f in os.listdir(frame_path) if f.lower().endswith(('.png', '.jpg'))]
                            if frame_files:
                                sample_frame = os.path.join(frame_path, frame_files[0])
                        
                        data.append({
                            'video_path': video_path,
                            'frame_dir': frame_path if os.path.exists(frame_path) else None,
                            'sample_frame': sample_frame,
                            'label': 0,  # 0 for real
                            'dataset': pattern['dataset_name'],
                            'method': 'original'
                        })
            
            # Process fake videos
            for fake_dir in pattern['fake_patterns']:
                if not os.path.exists(fake_dir):
                    continue
                    
                method_name = os.path.basename(fake_dir)
                for vid_file in tqdm(os.listdir(fake_dir), desc=f"Loading {pattern['dataset_name']} {method_name} videos"):
                    if vid_file.lower().endswith(('.mp4', '.avi', '.mov')):
                        # Extract image, audio, and video paths
                        video_path = os.path.join(fake_dir, vid_file)
                        
                        # Derive frame path (if frames are extracted separately)
                        frames_dir = fake_dir.replace('videos', 'frames')
                        frame_path = os.path.join(frames_dir, os.path.splitext(vid_file)[0])
                        
                        # Get a sample frame if possible
                        sample_frame = None
                        if os.path.exists(frame_path):
                            frame_files = [f for f in os.listdir(frame_path) if f.lower().endswith(('.png', '.jpg'))]
                            if frame_files:
                                sample_frame = os.path.join(frame_path, frame_files[0])
                        
                        data.append({
                            'video_path': video_path,
                            'frame_dir': frame_path if os.path.exists(frame_path) else None,
                            'sample_frame': sample_frame,
                            'label': 1,  # 1 for fake
                            'dataset': pattern['dataset_name'],
                            'method': method_name
                        })
        
        # Save to JSON for faster loading next time
        with open(json_path, 'w') as f:
            json.dump(data, f)
            
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Get dataset item
        
        Args:
            idx: Index
            
        Returns:
            Dictionary containing modality data and label
        """
        item = self.data[idx]
        label = item['label']
        
        result = {'label': label}
        
        try:
            # Process video data if available
            if 'video' in self.modalities and 'video_path' in item and os.path.exists(item['video_path']):
                video_data = self.video_processor.process_video(item['video_path'])
                result['video_frames'] = video_data['video_frames']
                
                # Also extract audio if needed
                if 'audio' in self.modalities:
                    result['audio_features'] = video_data['audio_features']
            
            # Process image data if available and video not already processed
            elif 'image' in self.modalities and 'sample_frame' in item and item['sample_frame'] and os.path.exists(item['sample_frame']):
                image = Image.open(item['sample_frame']).convert('RGB')
                
                # Apply transforms if specified
                if self.transform:
                    image = self.transform(image)
                    
                result['image'] = image
            
            # If neither video nor image was processed, create dummy tensors
            if 'video_frames' not in result and 'image' not in result:
                if 'video' in self.modalities:
                    result['video_frames'] = torch.zeros(self.num_frames, 3, self.image_size, self.image_size)
                    
                if 'image' in self.modalities:
                    result['image'] = torch.zeros(3, self.image_size, self.image_size)
            
            # Create dummy audio features if needed
            if 'audio' in self.modalities and 'audio_features' not in result:
                result['audio_features'] = {
                    'mel_spectrogram': torch.zeros(1, 128, 87),
                    'mfcc': torch.zeros(1, 120, 87)
                }
                
            return result
            
        except Exception as e:
            print(f"Error processing item {idx}: {e}")
            
            # Return dummy data
            result = {'label': label}
            
            if 'video' in self.modalities:
                result['video_frames'] = torch.zeros(self.num_frames, 3, self.image_size, self.image_size)
                
            if 'image' in self.modalities:
                result['image'] = torch.zeros(3, self.image_size, self.image_size)
                
            if 'audio' in self.modalities:
                result['audio_features'] = {
                    'mel_spectrogram': torch.zeros(1, 128, 87),
                    'mfcc': torch.zeros(1, 120, 87)
                }
                
            return result


class AddGaussianNoise:
    """
    Add Gaussian noise to tensor
    """
    def __init__(self, mean=0.0, std=0.01):
        self.mean = mean
        self.std = std
        
    def __call__(self, tensor):
        return tensor + torch.randn_like(tensor) * self.std + self.mean


class AddJPEGNoise:
    """
    Add JPEG compression artifacts
    """
    def __init__(self, quality_lower=70, quality_upper=90):
        self.quality_lower = quality_lower
        self.quality_upper = quality_upper
        
    def __call__(self, tensor):
        quality = random.randint(self.quality_lower, self.quality_upper)
        
        # Convert to PIL Image
        img = transforms.ToPILImage()(tensor)
        
        # Save with JPEG compression
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        
        # Load compressed image
        img = Image.open(buffer)
        
        # Convert back to tensor
        tensor = transforms.ToTensor()(img)
        
        return tensor


# # ml/data/data_utils.py
# import os
# import cv2
# import numpy as np
# from tqdm import tqdm
# from facenet_pytorch import MTCNN
# import torch
# import json
# import shutil


# def extract_frames(video_path, output_dir, max_frames=300, frame_interval=30):
#     """
#     Extract frames from a video file
    
#     Args:
#         video_path: Path to video file
#         output_dir: Directory to save extracted frames
#         max_frames: Maximum number of frames to extract
#         frame_interval: Extract every nth frame
        
#     Returns:
#         List of extracted frame paths
#     """
#     # Create output directory if it doesn't exist
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Open video file
#     cap = cv2.VideoCapture(video_path)
    
#     # Check if video opened successfully
#     if not cap.isOpened():
#         print(f"Error: Could not open video {video_path}")
#         return []
    
#     # Get video properties
#     frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     fps = cap.get(cv2.CAP_PROP_FPS)
    
#     # Calculate frame indices to extract
#     if frame_count <= max_frames * frame_interval:
#         # Extract every nth frame
#         frame_indices = list(range(0, frame_count, frame_interval))
#     else:
#         # Sample frames uniformly
#         frame_indices = np.linspace(0, frame_count - 1, max_frames, dtype=int)
    
#     # Extract frames
#     frame_paths = []
    
#     for i, frame_idx in enumerate(frame_indices):
#         # Set frame position
#         cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        
#         # Read frame
#         ret, frame = cap.read()
        
#         if ret:
#             # Save frame
#             frame_path = os.path.join(output_dir, f'frame_{i:04d}.png')
#             cv2.imwrite(frame_path, frame)
#             frame_paths.append(frame_path)
    
#     # Release video
#     cap.release()
    
#     return frame_paths


# def extract_faces(frame_paths, output_dir, face_size=224, confidence_threshold=0.9, device=None):
#     """
#     Extract faces from frames
    
#     Args:
#         frame_paths: List of frame paths
#         output_dir: Directory to save extracted faces
#         face_size: Size of output face images
#         confidence_threshold: Face detection confidence threshold
#         device: Device to run face detection on
        
#     Returns:
#         List of extracted face paths
#     """
#     # Create output directory if it doesn't exist
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Set device
#     if device is None:
#         device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
#     # Initialize face detector
#     face_detector = MTCNN(
#         image_size=face_size,
#         margin=0.2,
#         min_face_size=40,
#         thresholds=[0.6, 0.7, 0.9],
#         factor=0.85,
#         keep_all=True,
#         device=device
#     )
    
#     # Extract faces
#     face_paths = []
    
#     for frame_path in tqdm(frame_paths, desc="Extracting faces"):
#         # Load frame
#         frame = cv2.imread(frame_path)
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
#         # Detect faces
#         boxes, probs = face_detector.detect(frame_rgb)
        
#         # Skip if no faces detected
#         if boxes is None:
#             continue
        
#         # Extract each face
#         for i, (box, prob) in enumerate(zip(boxes, probs)):
#             # Skip if confidence is too low
#             if prob < confidence_threshold:
#                 continue
            
#             # Get coordinates
#             x1, y1, x2, y2 = [int(b) for b in box]
            
#             # Ensure box is within image bounds
#             h, w = frame.shape[:2]
#             x1, y1 = max(0, x1), max(0, y1)
#             x2, y2 = min(w, x2), min(h, y2)
            
#             # Skip if box is too small
#             if x2 - x1 < 50 or y2 - y1 < 50:
#                 continue
            
#             # Extract face
#             face = frame[y1:y2, x1:x2]
            
#             # Resize face
#             face = cv2.resize(face, (face_size, face_size))
            
#             # Save face
#             frame_name = os.path.basename(frame_path)
#             face_path = os.path.join(output_dir, f'{os.path.splitext(frame_name)[0]}_face_{i}.png')
#             cv2.imwrite(face_path, face)
#             face_paths.append(face_path)
    
#     return face_paths


# def prepare_dataset(source_dir, target_dir, dataset_name, split_ratio=(0.7, 0.15, 0.15)):
#     """
#     Prepare dataset for training, validation, and testing
    
#     Args:
#         source_dir: Source directory containing processed data
#         target_dir: Target directory to save organized dataset
#         dataset_name: Dataset name
#         split_ratio: Train/val/test split ratio
        
#     Returns:
#         Dictionary with dataset statistics
#     """
#     # Create target directories
#     os.makedirs(target_dir, exist_ok=True)
    
#     train_dir = os.path.join(target_dir, 'train')
#     val_dir = os.path.join(target_dir, 'val')
#     test_dir = os.path.join(target_dir, 'test')
    
#     os.makedirs(train_dir, exist_ok=True)
#     os.makedirs(val_dir, exist_ok=True)
#     os.makedirs(test_dir, exist_ok=True)
    
#     # Create class subdirectories
#     train_real_dir = os.path.join(train_dir, 'real')
#     train_fake_dir = os.path.join(train_dir, 'fake')
#     val_real_dir = os.path.join(val_dir, 'real')
#     val_fake_dir = os.path.join(val_dir, 'fake')
#     test_real_dir = os.path.join(test_dir, 'real')
#     test_fake_dir = os.path.join(test_dir, 'fake')
    
#     os.makedirs(train_real_dir, exist_ok=True)
#     os.makedirs(train_fake_dir, exist_ok=True)
#     os.makedirs(val_real_dir, exist_ok=True)
#     os.makedirs(val_fake_dir, exist_ok=True)
#     os.makedirs(test_real_dir, exist_ok=True)
#     os.makedirs(test_fake_dir, exist_ok=True)
    
#     # Process real data
#     real_source = os.path.join(source_dir, 'real_faces')
#     if not os.path.exists(real_source):
#         real_source = os.path.join(source_dir, 'real')
    
#     # Get all real face images
#     real_images = []
#     if os.path.exists(real_source):
#         for root, _, files in os.walk(real_source):
#             for file in files:
#                 if file.lower().endswith(('.png', '.jpg', '.jpeg')):
#                     real_images.append(os.path.join(root, file))
    
#     # Process fake data
#     fake_source = os.path.join(source_dir, 'fake_faces')
#     if not os.path.exists(fake_source):
#         fake_source = os.path.join(source_dir, 'fake')
    
#     # Get all fake face images
#     fake_images = []
#     if os.path.exists(fake_source):
#         for root, _, files in os.walk(fake_source):
#             for file in files:
#                 if file.lower().endswith(('.png', '.jpg', '.jpeg')):
#                     fake_images.append(os.path.join(root, file))
    
#     # Shuffle data
#     np.random.shuffle(real_images)
#     np.random.shuffle(fake_images)
    
#     # Calculate split indices
#     real_train_idx = int(len(real_images) * split_ratio[0])
#     real_val_idx = real_train_idx + int(len(real_images) * split_ratio[1])
    
#     fake_train_idx = int(len(fake_images) * split_ratio[0])
#     fake_val_idx = fake_train_idx + int(len(fake_images) * split_ratio[1])
    
#     # Split data
#     real_train = real_images[:real_train_idx]
#     real_val = real_images[real_train_idx:real_val_idx]
#     real_test = real_images[real_val_idx:]
    
#     fake_train = fake_images[:fake_train_idx]
#     fake_val = fake_images[fake_train_idx:fake_val_idx]
#     fake_test = fake_images[fake_val_idx:]
    
#     # Copy files
#     for src in tqdm(real_train, desc="Copying real train images"):
#         dst = os.path.join(train_real_dir, os.path.basename(src))
#         shutil.copy2(src, dst)
    
#     for src in tqdm(real_val, desc="Copying real val images"):
#         dst = os.path.join(val_real_dir, os.path.basename(src))
#         shutil.copy2(src, dst)
    
#     for src in tqdm(real_test, desc="Copying real test images"):
#         dst = os.path.join(test_real_dir, os.path.basename(src))
#         shutil.copy2(src, dst)
    
#     for src in tqdm(fake_train, desc="Copying fake train images"):
#         dst = os.path.join(train_fake_dir, os.path.basename(src))
#         shutil.copy2(src, dst)
    
#     for src in tqdm(fake_val, desc="Copying fake val images"):
#         dst = os.path.join(val_fake_dir, os.path.basename(src))
#         shutil.copy2(src, dst)
    
#     for src in tqdm(fake_test, desc="Copying fake test images"):
#         dst = os.path.join(test_fake_dir, os.path.basename(src))
#         shutil.copy2(src, dst)
    
#     # Create metadata
#     metadata = {
#         'dataset_name': dataset_name,
#         'statistics': {
#             'train': {
#                 'real': len(real_train),
#                 'fake': len(fake_train),
#                 'total': len(real_train) + len(fake_train)
#             },
#             'val': {
#                 'real': len(real_val),
#                 'fake': len(fake_val),
#                 'total': len(real_val) + len(fake_val)
#             },
#             'test': {
#                 'real': len(real_test),
#                 'fake': len(fake_test),
#                 'total': len(real_test) + len(fake_test)
#             },
#             'total': {
#                 'real': len(real_images),
#                 'fake': len(fake_images),
#                 'total': len(real_images) + len(fake_images)
#             }
#         }
#     }
    
#     # Save metadata
#     with open(os.path.join(target_dir, 'metadata.json'), 'w') as f:
#         json.dump(metadata, f, indent=4)
    
#     print(f"Dataset prepared successfully at {target_dir}")
    
#     return metadata


# def load_df40_dataset(dataset_path, subset='train'):
#     """
#     Load DF40 dataset
    
#     Args:
#         dataset_path: Path to DF40 dataset
#         subset: Dataset subset ('train', 'val', or 'test')
        
#     Returns:
#         Dictionary with dataset information
#     """
#     # Check if dataset exists
#     if not os.path.exists(dataset_path):
#         raise ValueError(f"Dataset path {dataset_path} does not exist")
    
#     # Load metadata
#     metadata_path = os.path.join(dataset_path, 'metadata.json')
#     if os.path.exists(metadata_path):
#         with open(metadata_path, 'r') as f:
#             metadata = json.load(f)
#     else:
#         metadata = None
    
#     # Setup paths
#     subset_path = os.path.join(dataset_path, subset)
#     if not os.path.exists(subset_path):
#         raise ValueError(f"Subset path {subset_path} does not exist")
    
#     real_path = os.path.join(subset_path, 'real')
#     fake_path = os.path.join(subset_path, 'fake')
    
#     # Get real and fake images
#     real_images = []
#     if os.path.exists(real_path):
#         for root, _, files in os.walk(real_path):
#             for file in files:
#                 if file.lower().endswith(('.png', '.jpg', '.jpeg')):
#                     real_images.append(os.path.join(root, file))
    
#     fake_images = []
#     if os.path.exists(fake_path):
#         for root, _, files in os.walk(fake_path):
#             for file in files:
#                 if file.lower().endswith(('.png', '.jpg', '.jpeg')):
#                     fake_images.append(os.path.join(root, file))
    
#     # Get methods and domains
#     methods = set()
#     domains = set()
    
#     for image_path in fake_images:
#         # Extract method and domain from path
#         parts = image_path.split(os.sep)
        
#         # Try to extract method and domain information
#         method = None
#         domain = None
        
#         for part in parts:
#             # Check if part contains method information
#             for method_name in ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures', 'FSGAN', 
#                                'SimSwap', 'FaceShifter', 'DeeperForensics', 'CDFI', 'FOMM',
#                                'StyleGAN', 'StarGAN', 'GauGAN', 'DDPM', 'DiT']:
#                 if method_name.lower() in part.lower():
#                     method = method_name
#                     break
            
#             # Check if part contains domain information
#             for domain_name in ['ff', 'cdf', 'dfdc', 'FF++', 'Celeb-DF']:
#                 if domain_name.lower() in part.lower():
#                     domain = domain_name
#                     break
        
#         if method:
#             methods.add(method)
        
#         if domain:
#             domains.add(domain)
    
#     return {
#         'metadata': metadata,
#         'real_images': real_images,
#         'fake_images': fake_images,
#         'methods': list(methods),
#         'domains': list(domains),
#         'total_images': len(real_images) + len(fake_images)
#     }