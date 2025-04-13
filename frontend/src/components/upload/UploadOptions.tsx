import React from 'react';
import Button from '../common/Button';

interface UploadOptionsProps {
    onSubmit: () => void;
    onCancel: () => void;
    onOptionChange: (name: string, value: any) => void;
    options: {
        detailedAnalysis: boolean;
        confidenceThreshold: number;
    };
    isProcessing: boolean;
    className?: string;
}

const UploadOptions: React.FC<UploadOptionsProps> = ({
    onSubmit,
    onCancel,
    onOptionChange,
    options,
    isProcessing,
    className = '',
}) => {
    return (
        <div className={`bg-gray-800 border border-gray-700 rounded-lg p-4 ${className}`}>
            <h3 className="text-lg font-medium text-white mb-4">Detection Options</h3>

            <div className="space-y-4">
                <div>
                    <label className="flex items-center space-x-3 cursor-pointer">
                        <input
                            type="checkbox"
                            className="form-checkbox h-5 w-5 text-blue-600 rounded focus:ring-blue-500 focus:ring-opacity-50 bg-gray-700 border-gray-600"
                            checked={options.detailedAnalysis}
                            onChange={(e) => onOptionChange('detailedAnalysis', e.target.checked)}
                            disabled={isProcessing}
                        />
                        <span className="text-gray-300">Perform detailed analysis</span>
                    </label>
                    <p className="text-gray-400 text-sm mt-1 ml-8">
                        Provides advanced visualizations and insights (takes longer)
                    </p>
                </div>

                <div>
                    <label className="block text-gray-300 mb-2">
                        Confidence threshold: {options.confidenceThreshold.toFixed(2)}
                    </label>
                    <input
                        type="range"
                        min="0"
                        max="1"
                        step="0.01"
                        value={options.confidenceThreshold}
                        onChange={(e) => onOptionChange('confidenceThreshold', parseFloat(e.target.value))}
                        disabled={isProcessing}
                        className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer"
                    />
                    <div className="flex justify-between text-xs text-gray-400 mt-1">
                        <span>0.00 (Lower threshold)</span>
                        <span>0.50</span>
                        <span>1.00 (Higher threshold)</span>
                    </div>
                </div>
            </div>

            <div className="mt-6 flex space-x-3">
                <Button
                    variant="primary"
                    onClick={onSubmit}
                    disabled={isProcessing}
                    className="flex-1"
                >
                    {isProcessing ? 'Processing...' : 'Detect Deepfake'}
                </Button>

                <Button
                    variant="secondary"
                    onClick={onCancel}
                    disabled={isProcessing}
                >
                    Cancel
                </Button>
            </div>
        </div>
    );
};

export default UploadOptions;