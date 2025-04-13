import React from 'react';
import ConfidenceGauge from '../common/ConfidenceGauge';

interface ConfidenceScoreProps {
    score: number;
    isFake: boolean;
    mediaType: 'image' | 'audio' | 'video';
    className?: string;
}

const ConfidenceScore: React.FC<ConfidenceScoreProps> = ({
    score,
    isFake,
    mediaType,
    className = '',
}) => {
    return (
        <div className={`bg-gray-800 rounded-lg p-4 border border-gray-700 ${className}`}>
            <h3 className="text-lg font-medium text-white mb-4">Detection Result</h3>

            <div className="flex flex-col items-center">
                <ConfidenceGauge score={score} size="lg" />

                <div className={`mt-4 text-center p-2 rounded-lg ${isFake
                        ? 'bg-red-900 bg-opacity-30 text-red-300'
                        : 'bg-green-900 bg-opacity-30 text-green-300'
                    }`}>
                    <span className="font-bold">
                        {isFake ? 'MANIPULATED' : 'AUTHENTIC'}
                    </span>
                    <p className="text-sm mt-1 opacity-80">
                        {isFake
                            ? `This ${mediaType} appears to contain manipulated content`
                            : `This ${mediaType} appears to be authentic`
                        }
                    </p>
                </div>

                <div className="mt-4 text-sm text-gray-400 text-center">
                    <p>
                        Our model is {Math.round(score * 100)}% confident in this classification.
                        {score > 0.9 || score < 0.1
                            ? ' This is a high-confidence result.'
                            : score > 0.7 || score < 0.3
                                ? ' This is a medium-confidence result.'
                                : ' This is a low-confidence result.'
                        }
                    </p>
                </div>
            </div>
        </div>
    );
};

export default ConfidenceScore;