import { apiRequest, uploadFileRequest } from './api';

interface DetectionParams {
    media_type: 'image' | 'audio' | 'video';
    detailed_analysis?: boolean;
    confidence_threshold?: number;
}

interface DetectionResponse {
    task_id: string;
    status: string;
    media_type: string;
    estimated_time: number;
}

interface StatusResponse {
    status: string;
    progress: number;
    message?: string;
    result_id?: string;
}

// Upload file and start detection
export const uploadFile = async (
    file: File,
    params: DetectionParams
): Promise<DetectionResponse> => {
    return uploadFileRequest<DetectionResponse>('/upload/upload', file, params);
};

// Check status of detection task
export const checkDetectionStatus = async (taskId: string): Promise<StatusResponse> => {
    return apiRequest<StatusResponse>(`/detection/status/${taskId}`);
};

// Get recent detection results
export const getRecentResults = async (limit: number = 10): Promise<any[]> => {
    return apiRequest<any[]>(`/detection/results?limit=${limit}`);
};

// Get specific detection result
export const getDetectionResult = async (resultId: string): Promise<any> => {
    return apiRequest<any>(`/detection/results/${resultId}`);
};