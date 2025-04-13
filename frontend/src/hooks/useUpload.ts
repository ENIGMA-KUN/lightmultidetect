import { useState, useCallback } from 'react';
import axios from 'axios';

interface UploadOptions {
  mediaType: 'image' | 'audio' | 'video' | 'auto';
  detailedAnalysis: boolean;
  confidenceThreshold: number;
}

interface UploadProgress {
  loaded: number;
  total: number;
  percentage: number;
}

interface UploadResult {
  taskId: string;
  status: string;
  mediaType: string;
  estimatedTime: number;
}

export const useUpload = () => {
  const [file, setFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const [uploadProgress, setUploadProgress] = useState<UploadProgress | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [uploadResult, setUploadResult] = useState<UploadResult | null>(null);

  // Reset upload state
  const resetUpload = useCallback(() => {
    setFile(null);
    setIsUploading(false);
    setUploadProgress(null);
    setUploadError(null);
    setUploadResult(null);
  }, []);

  // Handle file selection
  const selectFile = useCallback((selectedFile: File) => {
    setFile(selectedFile);
    setUploadError(null);
  }, []);

  // Upload file to backend
  const uploadFile = useCallback(async (options: UploadOptions) => {
    if (!file) {
      setUploadError('No file selected');
      return null;
    }

    setIsUploading(true);
    setUploadProgress(null);
    setUploadError(null);
    setUploadResult(null);

    // Create form data
    const formData = new FormData();
    formData.append('file', file);
    
    // Add detection parameters
    const detectionParams = {
      media_type: options.mediaType,
      detailed_analysis: options.detailedAnalysis,
      confidence_threshold: options.confidenceThreshold,
    };
    formData.append('detection_params', JSON.stringify(detectionParams));

    try {
      // Upload file with progress tracking
      const response = await axios.post('/api/v1/upload/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        onUploadProgress: (progressEvent) => {
          const total = progressEvent.total || 0;
          const loaded = progressEvent.loaded;
          const percentage = total ? Math.round((loaded * 100) / total) : 0;
          
          setUploadProgress({
            loaded,
            total,
            percentage,
          });
        },
      });

      setUploadResult(response.data);
      return response.data;
    } catch (error) {
      if (axios.isAxiosError(error) && error.response) {
        setUploadError(error.response.data.detail || 'Upload failed');
      } else {
        setUploadError('Upload failed: Network error');
      }
      return null;
    } finally {
      setIsUploading(false);
    }
  }, [file]);

  // Validate file type and size
  const validateFile = useCallback((file: File, acceptedTypes: string[], maxSize: number): string | null => {
    // Check file type
    if (!acceptedTypes.includes(file.type)) {
      return `File type not supported. Accepted types: ${acceptedTypes.join(', ')}`;
    }
    
    // Check file size
    if (file.size > maxSize) {
      return `File too large. Maximum size: ${(maxSize / (1024 * 1024)).toFixed(1)} MB`;
    }
    
    return null;
  }, []);

  return {
    file,
    isUploading,
    uploadProgress,
    uploadError,
    uploadResult,
    selectFile,
    uploadFile,
    validateFile,
    resetUpload,
  };
};