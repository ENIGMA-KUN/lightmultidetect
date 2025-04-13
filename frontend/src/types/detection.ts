// Type definitions for detection-related interfaces

// Detection request parameters
export interface DetectionParams {
    media_type: 'image' | 'audio' | 'video' | 'auto';
    detailed_analysis: boolean;
    confidence_threshold: number;
  }
  
  // Upload detection response
  export interface DetectionResponse {
    task_id: string;
    status: string;
    media_type: string;
    estimated_time: number;
  }
  
  // Task status response
  export interface TaskStatus {
    status: 'pending' | 'progress' | 'success' | 'failure';
    progress: number;
    message?: string;
    result_id?: string;
  }
  
  // Detection model information
  export interface ModelInfo {
    name: string;
    description: string;
    performance?: {
      accuracy: number;
      f1_score: number;
    };
  }
  
  // Region of interest in an image
  export interface DetectionRegion {
    x: number;
    y: number;
    width: number;
    height: number;
    confidence: number;
    label?: string;
  }
  
  // Detection analysis options
  export interface AnalysisOption {
    id: string;
    name: string;
    description: string;
  }
  
  // File upload progress
  export interface UploadProgress {
    loaded: number;
    total: number;
    percentage: number;
  }
  
  // Detection state in the context
  export interface DetectionState {
    isProcessing: boolean;
    progress: number;
    statusMessage: string;
    resultId: string | null;
    error: string | null;
  }