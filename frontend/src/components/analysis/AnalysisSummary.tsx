import React from 'react';
import ConfidenceGauge from '../common/ConfidenceGauge';
import Button from '../common/Button';

interface AnalysisSummaryProps {
  isComplete: boolean;
  isFake: boolean;
  confidenceScore: number;
  mediaType: 'image' | 'audio' | 'video';
  onViewDetails: () => void;
  onNewAnalysis: () => void;
  className?: string;
}

const AnalysisSummary: React.FC<AnalysisSummaryProps> = ({
  isComplete,
  isFake,
  confidenceScore,
  mediaType,
  onViewDetails,
  onNewAnalysis,
  className = '',
}) => {
  // Get summary message based on result
  const getSummaryMessage = () => {
    if (!isComplete) {
      return "Analysis in progress...";
    }
    
    if (isFake) {
      if (confidenceScore > 0.9) {
        return "This content is very likely manipulated.";
      } else if (confidenceScore > 0.7) {
        return "This content appears to be manipulated.";
      } else {
        return "This content might be manipulated.";
      }
    } else {
      if (confidenceScore < 0.1) {
        return "This content is very likely authentic.";
      } else if (confidenceScore < 0.3) {
        return "This content appears to be authentic.";
      } else {
        return "This content might be authentic.";
      }
    }
  };
  
  // Get icon based on result
  const getResultIcon = () => {
    if (!isComplete) {
      return (
        <div className="h-16 w-16 rounded-full bg-blue-900 bg-opacity-30 flex items-center justify-center animate-pulse">
          <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8 text-blue-500 animate-spin" viewBox="0 0 24 24" fill="none" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
          </svg>
        </div>
      );
    }
    
    if (isFake) {
      return (
        <div className="h-16 w-16 rounded-full bg-red-900 bg-opacity-30 flex items-center justify-center">
          <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8 text-red-500" viewBox="0 0 20 20" fill="currentColor">
            <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
          </svg>
        </div>
      );
    } else {
      return (
        <div className="h-16 w-16 rounded-full bg-green-900 bg-opacity-30 flex items-center justify-center">
          <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8 text-green-500" viewBox="0 0 20 20" fill="currentColor">
            <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
          </svg>
        </div>
      );
    }
  };
  
  return (
    <div className={`bg-gray-800 border border-gray-700 rounded-lg p-6 ${className}`}>
      <div className="flex flex-col items-center text-center">
        {getResultIcon()}
        
        <h2 className="text-2xl font-bold mt-4 text-white">
          {getSummaryMessage()}
        </h2>
        
        <p className="mt-2 text-gray-400">
          {isComplete ? (
            `Our models ${isFake ? 'detected' : 'did not detect'} manipulation in this ${mediaType} with ${Math.round(confidenceScore * 100)}% confidence.`
          ) : (
            "We're analyzing your content to determine if it has been manipulated."
          )}
        </p>
        
        {isComplete && (
          <div className="mt-6">
            <ConfidenceGauge score={confidenceScore} size="lg" />
            
            <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-3">
              <Button 
                variant="primary"
                onClick={onViewDetails}
              >
                View Detailed Analysis
              </Button>
              
              <Button
                variant="secondary"
                onClick={onNewAnalysis}
              >
                Analyze New Content
              </Button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default AnalysisSummary;