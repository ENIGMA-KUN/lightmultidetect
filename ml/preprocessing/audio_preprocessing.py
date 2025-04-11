# ml/preprocessing/audio_preprocessing.py
import os
import librosa
import numpy as np
from tqdm import tqdm
import soundfile as sf
from scipy import signal


class AudioPreprocessor:
    """
    Preprocessor for audio-based deepfake detection
    """
    def __init__(self, sample_rate=16000, n_mels=128, n_mfcc=40, max_duration=10):
        """
        Initialize preprocessor
        
        Args:
            sample_rate: Target sample rate for audio
            n_mels: Number of mel bands for spectrogram
            n_mfcc: Number of MFCC coefficients
            max_duration: Maximum duration in seconds
        """
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.max_samples = max_duration * sample_rate
    
    def load_audio(self, audio_path):
        """
        Load and preprocess audio file
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Preprocessed audio samples
        """
        # Load audio file
        audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        
        # Trim silence
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
    
    def extract_spectral_features(self, audio):
        """
        Extract spectral features from audio
        
        Args:
            audio: Audio samples
            
        Returns:
            Dictionary of spectral features
        """
        # Extract spectral centroid
        centroid = librosa.feature.spectral_centroid(
            y=audio,
            sr=self.sample_rate
        )
        
        # Extract spectral rolloff
        rolloff = librosa.feature.spectral_rolloff(
            y=audio,
            sr=self.sample_rate
        )
        
        # Extract zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio)
        
        # Extract RMS energy
        rms = librosa.feature.rms(y=audio)
        
        # Combine features
        features = np.concatenate([
            centroid,
            rolloff,
            zcr,
            rms
        ])
        
        # Normalize
        features = (features - features.mean()) / (features.std() + 1e-8)
        
        return features
    
    def process_audio(self, audio_path, output_dir=None, save_features=False):
        """
        Process audio file for deepfake detection
        
        Args:
            audio_path: Path to audio file
            output_dir: Directory to save processed data
            save_features: Whether to save extracted features
            
        Returns:
            Dictionary with processed audio data
        """
        # Load audio
        audio = self.load_audio(audio_path)
        
        # Extract features
        mel_spec = self.extract_mel_spectrogram(audio)
        mfcc = self.extract_mfcc(audio)
        spectral = self.extract_spectral_features(audio)
        
        # Save features if requested
        if save_features and output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save mel spectrogram
            mel_path = os.path.join(
                output_dir,
                f"{os.path.splitext(os.path.basename(audio_path))[0]}_mel.npy"
            )
            np.save(mel_path, mel_spec)
            
            # Save MFCC features
            mfcc_path = os.path.join(
                output_dir,
                f"{os.path.splitext(os.path.basename(audio_path))[0]}_mfcc.npy"
            )
            np.save(mfcc_path, mfcc)
            
            # Save spectral features
            spectral_path = os.path.join(
                output_dir,
                f"{os.path.splitext(os.path.basename(audio_path))[0]}_spectral.npy"
            )
            np.save(spectral_path, spectral)
        
        return {
            'mel_spectrogram': mel_spec,
            'mfcc': mfcc,
            'spectral': spectral
        }
    
    def process_directory(self, input_dir, output_dir, file_extensions=('.wav', '.mp3', '.ogg')):
        """
        Process all audio files in a directory
        
        Args:
            input_dir: Input directory
            output_dir: Output directory
            file_extensions: File extensions to process
            
        Returns:
            Dictionary with processing statistics
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all audio files
        audio_files = []
        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.lower().endswith(file_extensions):
                    audio_files.append(os.path.join(root, file))
        
        # Process audio files
        stats = {
            'total_files': len(audio_files),
            'processed_files': 0,
            'failed_files': 0
        }
        
        for audio_path in tqdm(audio_files, desc="Processing audio files"):
            try:
                # Create audio-specific output directory
                audio_output_dir = os.path.join(
                    output_dir,
                    os.path.splitext(os.path.basename(audio_path))[0]
                )
                
                # Process audio
                self.process_audio(
                    audio_path=audio_path,
                    output_dir=audio_output_dir,
                    save_features=True
                )
                
                # Update stats
                stats['processed_files'] += 1
                
            except Exception as e:
                print(f"Error processing audio {audio_path}: {e}")
                stats['failed_files'] += 1
        
        return stats 