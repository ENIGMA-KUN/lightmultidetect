import React, { createContext, useReducer, useContext } from 'react';

// Define types
interface DetectionState {
    isProcessing: boolean;
    progress: number;
    statusMessage: string;
    resultId: string | null;
    error: string | null;
}

type DetectionAction =
    | { type: 'START_PROCESSING' }
    | { type: 'UPDATE_PROGRESS', payload: { progress: number, message: string } }
    | { type: 'DETECTION_SUCCESS', payload: { resultId: string } }
    | { type: 'DETECTION_ERROR', payload: { error: string } }
    | { type: 'RESET' };

// Create context
const DetectionContext = createContext<{
    state: DetectionState;
    dispatch: React.Dispatch<DetectionAction>;
}>({
    state: {
        isProcessing: false,
        progress: 0,
        statusMessage: '',
        resultId: null,
        error: null,
    },
    dispatch: () => null,
});

// Reducer
const detectionReducer = (state: DetectionState, action: DetectionAction): DetectionState => {
    switch (action.type) {
        case 'START_PROCESSING':
            return {
                ...state,
                isProcessing: true,
                progress: 0,
                statusMessage: 'Starting detection...',
                resultId: null,
                error: null,
            };
        case 'UPDATE_PROGRESS':
            return {
                ...state,
                progress: action.payload.progress,
                statusMessage: action.payload.message,
            };
        case 'DETECTION_SUCCESS':
            return {
                ...state,
                isProcessing: false,
                progress: 100,
                statusMessage: 'Detection complete',
                resultId: action.payload.resultId,
                error: null,
            };
        case 'DETECTION_ERROR':
            return {
                ...state,
                isProcessing: false,
                progress: 0,
                statusMessage: 'Detection failed',
                resultId: null,
                error: action.payload.error,
            };
        case 'RESET':
            return {
                isProcessing: false,
                progress: 0,
                statusMessage: '',
                resultId: null,
                error: null,
            };
        default:
            return state;
    }
};

// Provider
export const DetectionProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
    const [state, dispatch] = useReducer(detectionReducer, {
        isProcessing: false,
        progress: 0,
        statusMessage: '',
        resultId: null,
        error: null,
    });

    return (
        <DetectionContext.Provider value={{ state, dispatch }}>
            {children}
        </DetectionContext.Provider>
    );
};

// Custom hook to use the context
export const useDetectionContext = () => useContext(DetectionContext);