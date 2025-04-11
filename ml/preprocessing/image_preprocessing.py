# ml/preprocessing/image_preprocessing.py
import cv2
import numpy as np
import torch
from torchvision import transforms
from facenet_pytorch import MTCNN
import face_alignment
from PIL import Image
import io


class ImagePreprocessor:
    """
    Preprocessor for image-based deepfake detection
    """
    def __init__(self, face_detector=None, face_size=224, use_gpu=torch.cuda.is_available()):
        self.face_size = face_size
        self.device = torch.device('cuda:0' if use_gpu else 'cpu')
        
        # Initialize face detector if not provided
        if face_detector is None:
            self.face_detector = MTCNN(
                image_size=face_size,
                margin=0.2,
                keep_all=True,
                device=self.device
            )
        else:
            self.face_detector = face_detector
            
        # Face alignment model
        self.face_alignment = face_alignment.FaceAlignment(
            face_alignment.LandmarksType._2D,
            device='cuda' if use_gpu else 'cpu'
        )
        
        # Image normalization
        self.transform = transforms.Compose([
            transforms.Resize((face_size, face_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
    def extract_face(self, image):
        """
        Extract faces from an image
        
        Args:
            image: Input image (numpy array or PIL Image)
            
        Returns:
            List of detected face tensors
        """
        # Convert numpy array to PIL Image if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
        # Detect faces
        boxes, probs = self.face_detector.detect(image)
        
        # No faces detected
        if boxes is None:
            return []
            
        # Process each detected face
        face_tensors = []
        for box in boxes:
            # Get coordinates
            x1, y1, x2, y2 = [int(b) for b in box]
            
            # Extract face region
            face = image.crop((x1, y1, x2, y2))
            
            # Transform to tensor
            face_tensor = self.transform(face)
            face_tensors.append(face_tensor)
            
        return face_tensors
    
    def process_image(self, image_data, return_largest=True):
        """
        Process image data for deepfake detection
        
        Args:
            image_data: Input image data (file-like object, numpy array, or PIL Image)
            return_largest: Whether to return only the largest face
            
        Returns:
            Processed face tensor(s)
        """
        # Handle different input types
        if isinstance(image_data, (bytes, io.BytesIO)):
            if isinstance(image_data, bytes):
                image_data = io.BytesIO(image_data)
            image = Image.open(image_data).convert('RGB')
        elif isinstance(image_data, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB))
        elif isinstance(image_data, Image.Image):
            image = image_data
        else:
            raise ValueError(f"Unsupported image data type: {type(image_data)}")
            
        # Extract faces
        face_tensors = self.extract_face(image)
        
        # Handle no faces
        if not face_tensors:
            # Return a blank face tensor as fallback
            blank = torch.zeros(3, self.face_size, self.face_size)
            return blank.unsqueeze(0)
            
        # Return largest face or all faces
        if return_largest and len(face_tensors) > 1:
            # Get face with largest area
            areas = [(tensor.size(1) * tensor.size(2)) for tensor in face_tensors]
            largest_idx = np.argmax(areas)
            return face_tensors[largest_idx].unsqueeze(0)
        else:
            return torch.stack(face_tensors)
    
    def extract_frequency_features(self, face_tensor):
        """
        Extract frequency domain features useful for deepfake detection
        
        Args:
            face_tensor: Normalized face tensor
            
        Returns:
            Frequency domain features
        """
        # Convert tensor to numpy image
        face_np = face_tensor.permute(1, 2, 0).cpu().numpy()
        face_np = (face_np * 255).astype(np.uint8)
        
        # Convert to grayscale
        gray = cv2.cvtColor(face_np, cv2.COLOR_RGB2GRAY)
        
        # Apply DCT transform
        dct = cv2.dct(np.float32(gray))
        
        # Extract frequency band energies
        h, w = dct.shape
        
        # Low frequency region
        low_freq = dct[:h//8, :w//8]
        low_energy = np.sum(np.abs(low_freq))
        
        # Mid frequency region
        mid_freq = dct[h//8:h//2, w//8:w//2]
        mid_energy = np.sum(np.abs(mid_freq))
        
        # High frequency region
        high_freq = dct[h//2:, w//2:]
        high_energy = np.sum(np.abs(high_freq))
        
        # Return normalized energies
        total_energy = low_energy + mid_energy + high_energy
        if total_energy > 0:
            features = np.array([
                low_energy / total_energy,
                mid_energy / total_energy,
                high_energy / total_energy
            ])
        else:
            features = np.array([0.33, 0.33, 0.33])
            
        return torch.tensor(features)


# ml/preprocessing/audio_preprocessing.py
import librosa
import numpy as np
import torch
from scipy import signal


class AudioPreprocessor:
    """
    Preprocessor for audio-based deepfake detection
    """
    def __init__(self, sample_rate=16000, n_mels=128, n_mfcc=40, max_duration=10):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.max_samples = max_duration * sample_rate
        
    def load_audio(self, audio_file, sr=None):
        """
        Load audio file and resample if necessary
        
        Args:
            audio_file: Path to audio file or audio data
            sr: Target sample rate (uses default if None)
            
        Returns:
            Loaded audio samples
        """
        if sr is None:
            sr = self.sample_rate
            
        # Handle different input types
        if isinstance(audio_file, str):
            audio, _ = librosa.load(audio_file, sr=sr, mono=True)
        elif isinstance(audio_file, np.ndarray):
            audio = audio_file
        else:
            audio, _ = librosa.load(audio_file, sr=sr, mono=True)
            
        # Trim leading/trailing silence
        audio, _ = librosa.effects.trim(audio, top_db=20)
        
        # Fix audio length
        if len(audio) > self.max_samples:
            # Take center segment
            start = len(audio) // 2 - self.max_samples // 2
            audio = audio[start:start + self.max_samples]
        elif len(audio) < self.max_samples:
            # Pad with silence
            padding = self.max_samples - len(audio)
            audio = np.pad(audio, (0, padding), mode='constant')
            
        return audio
    
    def extract_mel_spectrogram(self, audio):
        """
        Extract mel spectrogram from audio
        
        Args:
            audio: Audio samples
            
        Returns:
            Mel spectrogram
        """
        # Extract mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            hop_length=512,
            n_fft=2048
        )
        
        # Convert to log scale
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize
        log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-8)
        
        return log_mel
    
    def extract_mfcc(self, audio):
        """
        Extract MFCC features from audio
        
        Args:
            audio: Audio samples
            
        Returns:
            MFCC features
        """
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            hop_length=512,
            n_fft=2048
        )
        
        # Add deltas and delta-deltas
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        
        # Combine features
        features = np.concatenate([mfccs, delta_mfccs, delta2_mfccs])
        
        # Normalize
        features = (features - features.mean()) / (features.std() + 1e-8)
        
        return features
    
    def process_audio(self, audio_file):
        """
        Process audio file for deepfake detection
        
        Args:
            audio_file: Path to audio file or audio data
            
        Returns:
            Dictionary of processed features
        """
        # Load audio
        audio = self.load_audio(audio_file)
        
        # Extract features
        mel_spec = self.extract_mel_spectrogram(audio)
        mfcc = self.extract_mfcc(audio)
        
        # Convert to tensors
        mel_spec_tensor = torch.from_numpy(mel_spec).unsqueeze(0).float()
        mfcc_tensor = torch.from_numpy(mfcc).unsqueeze(0).float()
        
        return {
            'mel_spectrogram': mel_spec_tensor,
            'mfcc': mfcc_tensor
        }


# ml/preprocessing/video_preprocessing.py
import cv2
import numpy as np
import torch
import os
import tempfile
from PIL import Image

from ml.preprocessing.image_preprocessing import ImagePreprocessor
from ml.preprocessing.audio_preprocessing import AudioPreprocessor


class VideoPreprocessor:
    """
    Preprocessor for video-based deepfake detection
    """
    def __init__(self, face_size=224, num_frames=16, sample_rate=16000, use_gpu=torch.cuda.is_available()):
        self.face_size = face_size
        self.num_frames = num_frames
        self.sample_rate = sample_rate
        self.device = torch.device('cuda:0' if use_gpu else 'cpu')
        
        # Initialize sub-preprocessors
        self.image_processor = ImagePreprocessor(face_size=face_size, use_gpu=use_gpu)
        self.audio_processor = AudioPreprocessor(sample_rate=sample_rate)
        
    def process_video(self, video_path):
        """
        Process video for deepfake detection
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary of processed features
        """
        # Extract frames and audio
        frames = self.extract_frames(video_path)
        audio_features = self.extract_audio(video_path)
        
        # Process frames
        processed_frames = self.process_frames(frames)
        
        return {
            'video_frames': processed_frames,
            'audio_features': audio_features
        }
    
    def extract_frames(self, video_path):
        """
        Extract frames from video at uniform intervals
        
        Args:
            video_path: Path to video file
            
        Returns:
            List of extracted frames
        """
        # Open video file
        cap = cv2.VideoCapture(video_path)
        
        # Check if video opened successfully
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
            
        # Get video properties
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame indices to extract
        if frame_count <= self.num_frames:
            # If video has fewer frames than needed, duplicate frames
            indices = np.arange(frame_count)
            # Repeat indices to match required frame count
            indices = np.resize(indices, self.num_frames)
        else:
            # Sample frames uniformly
            indices = np.linspace(0, frame_count - 1, self.num_frames, dtype=int)
            
        # Extract frames
        frames = []
        for idx in indices:
            # Set frame position
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            
            # Read frame
            ret, frame = cap.read()
            
            if ret:
                # Convert to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            else:
                # If frame reading fails, add a blank frame
                blank_frame = np.zeros((self.face_size, self.face_size, 3), dtype=np.uint8)
                frames.append(blank_frame)
                
        # Release video
        cap.release()
        
        return frames
    
    def process_frames(self, frames):
        """
        Process extracted frames for deepfake detection
        
        Args:
            frames: List of video frames
            
        Returns:
            Tensor of processed frames
        """
        # Process each frame
        processed_frames = []
        for frame in frames:
            # Convert to PIL Image
            pil_frame = Image.fromarray(frame)
            
            # Extract face using image processor
            face_tensor = self.image_processor.process_image(pil_frame, return_largest=True)
            
            processed_frames.append(face_tensor)
            
        # Stack frames into a video tensor [batch, frames, channels, height, width]
        video_tensor = torch.stack(processed_frames).squeeze(1)
        
        return video_tensor
    
    def extract_audio(self, video_path):
        """
        Extract audio from video for deepfake detection
        
        Args:
            video_path: Path to video file
            
        Returns:
            Audio features
        """
        # Create temporary audio file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
            temp_audio_path = temp_audio.name
            
        try:
            # Extract audio using ffmpeg
            os.system(f'ffmpeg -y -i "{video_path}" -vn -acodec pcm_s16le -ar {self.sample_rate} -ac 1 "{temp_audio_path}" -hide_banner -loglevel error')
            
            # Process audio features
            audio_features = self.audio_processor.process_audio(temp_audio_path)
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
                
        return audio_features
    
    def extract_optical_flow(self, frames):
        """
        Extract optical flow between consecutive frames
        
        Args:
            frames: List of video frames
            
        Returns:
            Tensor of optical flow features
        """
        if len(frames) < 2:
            return None
            
        # Convert frames to grayscale
        gray_frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in frames]
        
        # Calculate optical flow for consecutive frames
        flows = []
        for i in range(len(gray_frames) - 1):
            # Calculate optical flow
            flow = cv2.calcOpticalFlowFarneback(
                gray_frames[i],
                gray_frames[i + 1],
                None,
                0.5,     # Pyramid scale
                3,       # Levels
                15,      # Window size
                3,       # Iterations
                5,       # Poly neighborhood
                1.2,     # Poly sigma
                0        # Flags
            )
            
            # Convert flow to RGB visualization
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
            hsv[..., 0] = angle * 180 / np.pi / 2
            hsv[..., 1] = 255
            hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
            rgb_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            
            flows.append(rgb_flow)
            
        # Process flow frames
        processed_flows = []
        for flow in flows:
            # Convert to PIL Image
            pil_flow = Image.fromarray(flow)
            
            # Resize to match face size
            pil_flow = pil_flow.resize((self.face_size, self.face_size))
            
            # Convert to tensor
            flow_tensor = torch.from_numpy(np.array(pil_flow)).float() / 255.0
            flow_tensor = flow_tensor.permute(2, 0, 1)
            
            processed_flows.append(flow_tensor)
            
        # Stack flows into a tensor
        if processed_flows:
            flow_tensor = torch.stack(processed_flows)
        else:
            flow_tensor = None
            
        return flow_tensor