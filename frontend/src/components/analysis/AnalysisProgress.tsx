import React from 'react';
import ProgressIndicator from '../common/ProgressIndicator';

interface AnalysisProgressProps {
    progress: number;
    message: string;
    mediaType: 'image' | 'audio' | 'video';
    className?: string;
}

const AnalysisProgress: React.FC<AnalysisProgressProps> = ({
    progress,
    message,
    mediaType,
    className = '',
}) => {
    return (
        <div className={`bg-gray-800 border border-gray-700 rounded-lg p-6 ${className}`}>
            <div className="flex items-center mb-4">
                <div className="mr-4">
                    {mediaType === 'image' && (
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-10 w-10 text-blue-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                        </svg>
                    )}
                    {mediaType === 'audio' && (
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-10 w-10 text-blue-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3" />
                        </svg>
                    )}
                    {mediaType === 'video' && (
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-10 w-10 text-blue-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                        </svg>
                    )}
                </div>
                <div>
                    <h3 className="text-xl font-medium text-white">Analyzing {mediaType}</h3>
                    <p className="text-gray-300">{message}</p>
                </div>
            </div>

            <ProgressIndicator
                progress={progress}
                size="lg"
                className="mb-6"
            />

            {/* Processing steps */}
            <div className="space-y-2">
                <ProcessingStep
                    title="Preprocessing"
                    isActive={progress < 30}
                    isComplete={progress >= 30}
                />
                <ProcessingStep
                    title="Running detection models"
                    isActive={progress >= 30 && progress < 60}
                    isComplete={progress >= 60}
                />
                <ProcessingStep
                    title="Analyzing results"
                    isActive={progress >= 60 && progress < 80}
                    isComplete={progress >= 80}
                />
                <ProcessingStep
                    title="Generating visualizations"
                    isActive={progress >= 80 && progress < 95}
                    isComplete={progress >= 95}
                />
            </div>

            <div className="mt-6 text-center text-sm text-gray-400">
                <p>Processing time may vary depending on file size and complexity</p>
            </div>
        </div>
    );
};

interface ProcessingStepProps {
    title: string;
    isActive: boolean;
    isComplete: boolean;
}

const ProcessingStep: React.FC<ProcessingStepProps> = ({ title, isActive, isComplete }) => {
    return (
        <div className={`flex items-center p-2 rounded-md transition-colors duration-300 ${isActive ? 'bg-blue-900 bg-opacity-30' : isComplete ? 'bg-green-900 bg-opacity-20' : 'bg-gray-900 bg-opacity-50'
            }`}>
            <div className="mr-3">
                {isComplete ? (
                    <div className="h-6 w-6 rounded-full bg-green-500 flex items-center justify-center">
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 text-white" viewBox="0 0 20 20" fill="currentColor">
                            <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                        </svg>
                    </div>
                ) : isActive ? (
                    <div className="h-6 w-6 rounded-full bg-blue-500 flex items-center justify-center animate-pulse">
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 text-white animate-spin" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                        </svg>
                    </div>
                ) : (
                    <div className="h-6 w-6 rounded-full bg-gray-700 flex items-center justify-center">
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 text-gray-400" viewBox="0 0 20 20" fill="currentColor">
                            <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm1-12a1 1 0 10-2 0v4a1 1 0 00.293.707l2.828 2.829a1 1 0 101.415-1.415L11 9.586V6z" clipRule="evenodd" />
                        </svg>
                    </div>
                )}
            </div>
            <div className="text-gray-300">{title}</div>
        </div>
    );
};

export default AnalysisProgress;