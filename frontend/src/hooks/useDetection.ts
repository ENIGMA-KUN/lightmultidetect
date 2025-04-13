import { useState, useCallback, useRef } from 'react';
import { useDetectionContext } from '../contexts/DetectionContext';
import { uploadFile, checkDetectionStatus } from '../services/detection';

export const useDetection = () => {
    const { state, dispatch } = useDetectionContext();
    const { isProcessing, progress, statusMessage, resultId, error } = state;

    const [taskId, setTaskId] = useState<string | null>(null);
    const statusCheckRef = useRef<number | null>(null);

    // Start detection process
    const startDetection = useCallback(async (
        file: File,
        mediaType: 'image' | 'audio' | 'video',
        detailedAnalysis: boolean,
        confidenceThreshold: number
    ) => {
        // Clear any existing interval
        if (statusCheckRef.current) {
            window.clearInterval(statusCheckRef.current);
        }

        dispatch({ type: 'START_PROCESSING' });

        try {
            // Upload file and start detection
            const response = await uploadFile(file, {
                media_type: mediaType,
                detailed_analysis: detailedAnalysis,
                confidence_threshold: confidenceThreshold,
            });

            if (response.task_id) {
                setTaskId(response.task_id);
                dispatch({
                    type: 'UPDATE_PROGRESS',
                    payload: {
                        progress: 20,
                        message: 'Processing media...'
                    }
                });

                // Start polling for status
                statusCheckRef.current = window.setInterval(() => {
                    checkStatus(response.task_id);
                }, 2000);
            } else {
                throw new Error('No task ID returned from server');
            }
        } catch (error) {
            dispatch({
                type: 'DETECTION_ERROR',
                payload: { error: error instanceof Error ? error.message : 'Unknown error' }
            });
        }
    }, [dispatch]);

    // Check status of detection task
    const checkStatus = useCallback(async (task_id: string) => {
        try {
            const statusResponse = await checkDetectionStatus(task_id);

            if (statusResponse.status === 'pending' || statusResponse.status === 'progress') {
                dispatch({
                    type: 'UPDATE_PROGRESS',
                    payload: {
                        progress: statusResponse.progress || 0,
                        message: statusResponse.message || 'Processing...'
                    }
                });
            } else if (statusResponse.status === 'success' && statusResponse.result_id) {
                // Stop polling
                if (statusCheckRef.current) {
                    window.clearInterval(statusCheckRef.current);
                    statusCheckRef.current = null;
                }

                dispatch({
                    type: 'DETECTION_SUCCESS',
                    payload: { resultId: statusResponse.result_id }
                });
            } else if (statusResponse.status === 'failure') {
                // Stop polling
                if (statusCheckRef.current) {
                    window.clearInterval(statusCheckRef.current);
                    statusCheckRef.current = null;
                }

                dispatch({
                    type: 'DETECTION_ERROR',
                    payload: { error: statusResponse.message || 'Detection failed' }
                });
            }
        } catch (error) {
            // Don't stop polling on network errors, just log
            console.error('Error checking status:', error);
        }
    }, [dispatch]);

    // Reset detection state
    const resetDetection = useCallback(() => {
        if (statusCheckRef.current) {
            window.clearInterval(statusCheckRef.current);
            statusCheckRef.current = null;
        }

        dispatch({ type: 'RESET' });
    }, [dispatch]);

    return {
        isProcessing,
        progress,
        statusMessage,
        resultId,
        error,
        taskId,
        startDetection,
        checkDetectionStatus: checkStatus,
        resetDetection,
    };
};