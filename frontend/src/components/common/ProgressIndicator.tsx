import React from 'react';

interface ProgressIndicatorProps {
    progress: number;
    message?: string;
    showLabel?: boolean;
    size?: 'sm' | 'md' | 'lg';
    className?: string;
}

const ProgressIndicator: React.FC<ProgressIndicatorProps> = ({
    progress,
    message,
    showLabel = true,
    size = 'md',
    className = '',
}) => {
    // Cap progress between 0 and 100
    const safeProgress = Math.min(100, Math.max(0, progress));

    // Size classes
    const sizeClasses = {
        sm: 'h-1',
        md: 'h-2',
        lg: 'h-4',
    };

    // Determine color based on progress
    let colorClass = 'bg-blue-500';
    if (safeProgress < 30) {
        colorClass = 'bg-red-500';
    } else if (safeProgress < 70) {
        colorClass = 'bg-yellow-500';
    } else {
        colorClass = 'bg-green-500';
    }

    return (
        <div className={`w-full ${className}`}>
            {showLabel && (
                <div className="flex justify-between mb-1">
                    <span className="text-sm font-medium text-gray-200">
                        {message || 'Progress'}
                    </span>
                    <span className="text-sm font-medium text-gray-200">
                        {Math.round(safeProgress)}%
                    </span>
                </div>
            )}

            <div className={`w-full bg-gray-700 rounded-full overflow-hidden ${sizeClasses[size]}`}>
                <div
                    className={`${colorClass} rounded-full transition-all duration-300 ease-out`}
                    style={{ width: `${safeProgress}%` }}
                >
                    <div className="h-full w-full bg-opacity-30 bg-white bg-stripe-animate"></div>
                </div>
            </div>

            {message && !showLabel && (
                <div className="mt-1">
                    <span className="text-sm text-gray-400">{message}</span>
                </div>
            )}
        </div>
    );
};

export default ProgressIndicator;