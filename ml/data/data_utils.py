# ml/data/data_utils.py
import os
import cv2
import numpy as np
from tqdm import tqdm
from facenet_pytorch import MTCNN
import torch
import json
import shutil


def extract_frames(video_path, output_dir, max_frames=300, frame_interval=30):
    """
    Extract frames from a video file
    
    Args:
        video_path: Path to video file
        output_dir: Directory to save extracted frames
        max_frames: Maximum number of frames to extract
        frame_interval: Extract every nth frame
        
    Returns:
        List of extracted frame paths
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return []
    
    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Calculate frame indices to extract
    if frame_count <= max_frames * frame_interval:
        # Extract every nth frame
        frame_indices = list(range(0, frame_count, frame_interval))
    else:
        # Sample frames uniformly
        frame_indices = np.linspace(0, frame_count - 1, max_frames, dtype=int)
    
    # Extract frames
    frame_paths = []
    
    for i, frame_idx in enumerate(frame_indices):
        # Set frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        
        # Read frame
        ret, frame = cap.read()
        
        if ret:
            # Save frame
            frame_path = os.path.join(output_dir, f'frame_{i:04d}.png')
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
    
    # Release video
    cap.release()
    
    return frame_paths


def extract_faces(frame_paths, output_dir, face_size=224, confidence_threshold=0.9, device=None):
    """
    Extract faces from frames
    
    Args:
        frame_paths: List of frame paths
        output_dir: Directory to save extracted faces
        face_size: Size of output face images
        confidence_threshold: Face detection confidence threshold
        device: Device to run face detection on
        
    Returns:
        List of extracted face paths
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Initialize face detector
    face_detector = MTCNN(
        image_size=face_size,
        margin=0.2,
        min_face_size=40,
        thresholds=[0.6, 0.7, 0.9],
        factor=0.85,
        keep_all=True,
        device=device
    )
    
    # Extract faces
    face_paths = []
    
    for frame_path in tqdm(frame_paths, desc="Extracting faces"):
        # Load frame
        frame = cv2.imread(frame_path)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        boxes, probs = face_detector.detect(frame_rgb)
        
        # Skip if no faces detected
        if boxes is None:
            continue
        
        # Extract each face
        for i, (box, prob) in enumerate(zip(boxes, probs)):
            # Skip if confidence is too low
            if prob < confidence_threshold:
                continue
            
            # Get coordinates
            x1, y1, x2, y2 = [int(b) for b in box]
            
            # Ensure box is within image bounds
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            # Skip if box is too small
            if x2 - x1 < 50 or y2 - y1 < 50:
                continue
            
            # Extract face
            face = frame[y1:y2, x1:x2]
            
            # Resize face
            face = cv2.resize(face, (face_size, face_size))
            
            # Save face
            frame_name = os.path.basename(frame_path)
            face_path = os.path.join(output_dir, f'{os.path.splitext(frame_name)[0]}_face_{i}.png')
            cv2.imwrite(face_path, face)
            face_paths.append(face_path)
    
    return face_paths


def prepare_dataset(source_dir, target_dir, dataset_name, split_ratio=(0.7, 0.15, 0.15)):
    """
    Prepare dataset for training, validation, and testing
    
    Args:
        source_dir: Source directory containing processed data
        target_dir: Target directory to save organized dataset
        dataset_name: Dataset name
        split_ratio: Train/val/test split ratio
        
    Returns:
        Dictionary with dataset statistics
    """
    # Create target directories
    os.makedirs(target_dir, exist_ok=True)
    
    train_dir = os.path.join(target_dir, 'train')
    val_dir = os.path.join(target_dir, 'val')
    test_dir = os.path.join(target_dir, 'test')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Create class subdirectories
    train_real_dir = os.path.join(train_dir, 'real')
    train_fake_dir = os.path.join(train_dir, 'fake')
    val_real_dir = os.path.join(val_dir, 'real')
    val_fake_dir = os.path.join(val_dir, 'fake')
    test_real_dir = os.path.join(test_dir, 'real')
    test_fake_dir = os.path.join(test_dir, 'fake')
    
    os.makedirs(train_real_dir, exist_ok=True)
    os.makedirs(train_fake_dir, exist_ok=True)
    os.makedirs(val_real_dir, exist_ok=True)
    os.makedirs(val_fake_dir, exist_ok=True)
    os.makedirs(test_real_dir, exist_ok=True)
    os.makedirs(test_fake_dir, exist_ok=True)
    
    # Process real data
    real_source = os.path.join(source_dir, 'real_faces')
    if not os.path.exists(real_source):
        real_source = os.path.join(source_dir, 'real')
    
    # Get all real face images
    real_images = []
    if os.path.exists(real_source):
        for root, _, files in os.walk(real_source):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    real_images.append(os.path.join(root, file))
    
    # Process fake data
    fake_source = os.path.join(source_dir, 'fake_faces')
    if not os.path.exists(fake_source):
        fake_source = os.path.join(source_dir, 'fake')
    
    # Get all fake face images
    fake_images = []
    if os.path.exists(fake_source):
        for root, _, files in os.walk(fake_source):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    fake_images.append(os.path.join(root, file))
    
    # Shuffle data
    np.random.shuffle(real_images)
    np.random.shuffle(fake_images)
    
    # Calculate split indices
    real_train_idx = int(len(real_images) * split_ratio[0])
    real_val_idx = real_train_idx + int(len(real_images) * split_ratio[1])
    
    fake_train_idx = int(len(fake_images) * split_ratio[0])
    fake_val_idx = fake_train_idx + int(len(fake_images) * split_ratio[1])
    
    # Split data
    real_train = real_images[:real_train_idx]
    real_val = real_images[real_train_idx:real_val_idx]
    real_test = real_images[real_val_idx:]
    
    fake_train = fake_images[:fake_train_idx]
    fake_val = fake_images[fake_train_idx:fake_val_idx]
    fake_test = fake_images[fake_val_idx:]
    
    # Copy files
    for src in tqdm(real_train, desc="Copying real train images"):
        dst = os.path.join(train_real_dir, os.path.basename(src))
        shutil.copy2(src, dst)
    
    for src in tqdm(real_val, desc="Copying real val images"):
        dst = os.path.join(val_real_dir, os.path.basename(src))
        shutil.copy2(src, dst)
    
    for src in tqdm(real_test, desc="Copying real test images"):
        dst = os.path.join(test_real_dir, os.path.basename(src))
        shutil.copy2(src, dst)
    
    for src in tqdm(fake_train, desc="Copying fake train images"):
        dst = os.path.join(train_fake_dir, os.path.basename(src))
        shutil.copy2(src, dst)
    
    for src in tqdm(fake_val, desc="Copying fake val images"):
        dst = os.path.join(val_fake_dir, os.path.basename(src))
        shutil.copy2(src, dst)
    
    for src in tqdm(fake_test, desc="Copying fake test images"):
        dst = os.path.join(test_fake_dir, os.path.basename(src))
        shutil.copy2(src, dst)
    
    # Create metadata
    metadata = {
        'dataset_name': dataset_name,
        'statistics': {
            'train': {
                'real': len(real_train),
                'fake': len(fake_train),
                'total': len(real_train) + len(fake_train)
            },
            'val': {
                'real': len(real_val),
                'fake': len(fake_val),
                'total': len(real_val) + len(fake_val)
            },
            'test': {
                'real': len(real_test),
                'fake': len(fake_test),
                'total': len(real_test) + len(fake_test)
            },
            'total': {
                'real': len(real_images),
                'fake': len(fake_images),
                'total': len(real_images) + len(fake_images)
            }
        }
    }
    
    # Save metadata
    with open(os.path.join(target_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print(f"Dataset prepared successfully at {target_dir}")
    
    return metadata


def load_df40_dataset(dataset_path, subset='train'):
    """
    Load DF40 dataset
    
    Args:
        dataset_path: Path to DF40 dataset
        subset: Dataset subset ('train', 'val', or 'test')
        
    Returns:
        Dictionary with dataset information
    """
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        raise ValueError(f"Dataset path {dataset_path} does not exist")
    
    # Load metadata
    metadata_path = os.path.join(dataset_path, 'metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = None
    
    # Setup paths
    subset_path = os.path.join(dataset_path, subset)
    if not os.path.exists(subset_path):
        raise ValueError(f"Subset path {subset_path} does not exist")
    
    real_path = os.path.join(subset_path, 'real')
    fake_path = os.path.join(subset_path, 'fake')
    
    # Get real and fake images
    real_images = []
    if os.path.exists(real_path):
        for root, _, files in os.walk(real_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    real_images.append(os.path.join(root, file))
    
    fake_images = []
    if os.path.exists(fake_path):
        for root, _, files in os.walk(fake_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    fake_images.append(os.path.join(root, file))
    
    # Get methods and domains
    methods = set()
    domains = set()
    
    for image_path in fake_images:
        # Extract method and domain from path
        parts = image_path.split(os.sep)
        
        # Try to extract method and domain information
        method = None
        domain = None
        
        for part in parts:
            # Check if part contains method information
            for method_name in ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures', 'FSGAN', 
                               'SimSwap', 'FaceShifter', 'DeeperForensics', 'CDFI', 'FOMM',
                               'StyleGAN', 'StarGAN', 'GauGAN', 'DDPM', 'DiT']:
                if method_name.lower() in part.lower():
                    method = method_name
                    break
            
            # Check if part contains domain information
            for domain_name in ['ff', 'cdf', 'dfdc', 'FF++', 'Celeb-DF']:
                if domain_name.lower() in part.lower():
                    domain = domain_name
                    break
        
        if method:
            methods.add(method)
        
        if domain:
            domains.add(domain)
    
    return {
        'metadata': metadata,
        'real_images': real_images,
        'fake_images': fake_images,
        'methods': list(methods),
        'domains': list(domains),
        'total_images': len(real_images) + len(fake_images)
    } 