import React from 'react';
import Button from '../common/Button';

interface AnalysisOptionsProps {
    onModelChange: (model: string) => void;
    onViewChange: (view: string) => void;
    selectedModel: string;
    selectedView: string;
    modelOptions: Array<{ id: string; name: string; description: string }>;
    viewOptions: Array<{ id: string; name: string; description: string }>;
    className?: string;
}

const AnalysisOptions: React.FC<AnalysisOptionsProps> = ({
    onModelChange,
    onViewChange,
    selectedModel,
    selectedView,
    modelOptions,
    viewOptions,
    className = '',
}) => {
    return (
        <div className={`bg-gray-800 border border-gray-700 rounded-lg p-6 ${className}`}>
            <h3 className="text-lg font-medium text-white mb-4">Analysis Options</h3>

            {/* Model Selection */}
            <div className="mb-6">
                <label className="block text-gray-300 mb-2">Detection Model</label>
                <div className="grid grid-cols-1 gap-3">
                    {modelOptions.map((model) => (
                        <div
                            key={model.id}
                            onClick={() => onModelChange(model.id)}
                            className={`
                border rounded-lg p-3 cursor-pointer transition-all duration-200
                ${selectedModel === model.id
                                    ? 'border-blue-500 bg-blue-900 bg-opacity-20'
                                    : 'border-gray-600 hover:border-gray-400'}
              `}
                        >
                            <div className="flex items-center">
                                <div className={`h-4 w-4 rounded-full mr-3 ${selectedModel === model.id ? 'bg-blue-500' : 'bg-gray-600'}`} />
                                <div className="font-medium text-white">{model.name}</div>
                            </div>
                            <div className="mt-1 text-sm text-gray-400 ml-7">{model.description}</div>
                        </div>
                    ))}
                </div>
            </div>

            {/* View Selection */}
            <div>
                <label className="block text-gray-300 mb-2">Result View</label>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                    {viewOptions.map((view) => (
                        <div
                            key={view.id}
                            onClick={() => onViewChange(view.id)}
                            className={`
                border rounded-lg p-3 cursor-pointer transition-all duration-200
                ${selectedView === view.id
                                    ? 'border-purple-500 bg-purple-900 bg-opacity-20'
                                    : 'border-gray-600 hover:border-gray-400'}
              `}
                        >
                            <div className="flex items-center">
                                <div className={`h-4 w-4 rounded-full mr-3 ${selectedView === view.id ? 'bg-purple-500' : 'bg-gray-600'}`} />
                                <div className="font-medium text-white">{view.name}</div>
                            </div>
                            <div className="mt-1 text-sm text-gray-400 ml-7">{view.description}</div>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
};

export default AnalysisOptions;