import React, { useState } from 'react';
import ConfidenceScore from './ConfidenceScore';
import HeatmapView from './HeatmapView';
import TemporalAnalysis from './TemporalAnalysis';
import FrequencyAnalysis from './FrequencyAnalysis';
import ExplanationPanel from './ExplanationPanel';
import Button from '../common/Button';

interface ResultsPageProps {
    result: any; // Replace with proper type
    mediaType: 'image' | 'audio' | 'video';
    onNewAnalysis: () => void;
    className?: string;
}

const ResultsPage: React.FC<ResultsPageProps> = ({
    result,
    mediaType,
    onNewAnalysis,
    className = '',
}) => {
    const [activeTab, setActiveTab] = useState<'summary' | 'visual' | 'technical'>('summary');

    return (
        <div className={`bg-gray-900 rounded-lg border border-gray-700 shadow-xl overflow-hidden ${className}`}>
            {/* Header */}
            <div className="bg-gradient-to-r from-gray-800 to-gray-900 p-4 border-b border-gray-700">
                <div className="flex flex-wrap items-center justify-between">
                    <h2 className="text-xl font-bold text-white flex items-center">
                        <span className={`h-3 w-3 rounded-full mr-2 ${result.is_fake ? 'bg-red-500' : 'bg-green-500'}`}></span>
                        {result.is_fake ? 'Manipulated Content Detected' : 'Content Appears Authentic'}
                    </h2>

                    <div className="flex space-x-2 mt-2 sm:mt-0">
                        <Button
                            variant="secondary"
                            size="sm"
                            onClick={onNewAnalysis}
                        >
                            New Analysis
                        </Button>
                    </div>
                </div>
            </div>

            {/* Tabs */}
            <div className="bg-gray-800 border-b border-gray-700">
                <nav className="flex overflow-x-auto">
                    <button
                        onClick={() => setActiveTab('summary')}
                        className={`px-4 py-3 text-sm font-medium whitespace-nowrap transition-colors duration-200 focus:outline-none
              ${activeTab === 'summary'
                                ? 'text-blue-500 border-b-2 border-blue-500'
                                : 'text-gray-400 hover:text-gray-200'}
            `}
                    >
                        Summary
                    </button>

                    <button
                        onClick={() => setActiveTab('visual')}
                        className={`px-4 py-3 text-sm font-medium whitespace-nowrap transition-colors duration-200 focus:outline-none
              ${activeTab === 'visual'
                                ? 'text-blue-500 border-b-2 border-blue-500'
                                : 'text-gray-400 hover:text-gray-200'}
            `}
                    >
                        Visual Analysis
                    </button>

                    <button
                        onClick={() => setActiveTab('technical')}
                        className={`px-4 py-3 text-sm font-medium whitespace-nowrap transition-colors duration-200 focus:outline-none
              ${activeTab === 'technical'
                                ? 'text-blue-500 border-b-2 border-blue-500'
                                : 'text-gray-400 hover:text-gray-200'}
            `}
                    >
                        Technical Details
                    </button>
                </nav>
            </div>

            {/* Content */}
            <div className="p-4">
                {activeTab === 'summary' && (
                    <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
                        <div className="lg:col-span-1">
                            <ConfidenceScore
                                score={result.confidence_score}
                                isFake={result.is_fake}
                                mediaType={mediaType}
                            />
                        </div>

                        <div className="lg:col-span-2">
                            <ExplanationPanel
                                result={result}
                                mediaType={mediaType}
                            />
                        </div>
                    </div>
                )}

                {activeTab === 'visual' && (
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        {(mediaType === 'image' || mediaType === 'video') && result.visualizations?.heatmap && (
                            <div className="md:col-span-1">
                                <HeatmapView
                                    heatmapUrl={result.visualizations.heatmap.url}
                                    regions={result.visualizations.heatmap.regions}
                                />
                            </div>
                        )}

                        {(mediaType === 'audio' || mediaType === 'video') && result.visualizations?.temporal && (
                            <div className="md:col-span-1">
                                <TemporalAnalysis
                                    data={result.visualizations.temporal}
                                    mediaType={mediaType}
                                />
                            </div>
                        )}

                        {result.visualizations?.frequency && (
                            <div className="md:col-span-1">
                                <FrequencyAnalysis
                                    data={result.visualizations.frequency}
                                />
                            </div>
                        )}
                    </div>
                )}

                {activeTab === 'technical' && (
                    <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
                        <h3 className="text-lg font-medium text-white mb-3">Technical Details</h3>

                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            {/* Models Used */}
                            <div className="bg-gray-900 p-4 rounded-lg">
                                <h4 className="text-gray-300 font-medium mb-2">Models Used</h4>
                                <div className="space-y-2">
                                    {Object.entries(result.models_used || {}).map(([key, value]: [string, any]) => (
                                        <div key={key} className="flex justify-between">
                                            <span className="text-gray-400">{key}:</span>
                                            <span className="text-gray-300">{value}</span>
                                        </div>
                                    ))}
                                </div>
                            </div>

                            {/* Detection Scores */}
                            <div className="bg-gray-900 p-4 rounded-lg">
                                <h4 className="text-gray-300 font-medium mb-2">Detection Scores</h4>
                                <div className="space-y-2">
                                    {result.detection_details && result.detection_details.model_scores &&
                                        Object.entries(result.detection_details.model_scores).map(([key, value]: [string, any]) => (
                                            <div key={key} className="flex justify-between items-center">
                                                <span className="text-gray-400">{key}:</span>
                                                <div className="flex items-center">
                                                    <div className="w-32 h-2 bg-gray-700 rounded-full mr-2">
                                                        <div
                                                            className={`h-2 rounded-full ${value > 0.5 ? 'bg-red-500' : 'bg-green-500'}`}
                                                            style={{ width: `${value * 100}%` }}
                                                        ></div>
                                                    </div>
                                                    <span className="text-gray-300">{Math.round(value * 100)}%</span>
                                                </div>
                                            </div>
                                        ))
                                    }
                                </div>
                            </div>

                            {/* Additional Details */}
                            {result.detection_details && (
                                <div className="bg-gray-900 p-4 rounded-lg md:col-span-2">
                                    <h4 className="text-gray-300 font-medium mb-2">Additional Analysis</h4>
                                    <pre className="text-gray-400 text-sm overflow-auto p-2 bg-black bg-opacity-30 rounded max-h-60">
                                        {JSON.stringify(result.detection_details, null, 2)}
                                    </pre>
                                </div>
                            )}
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
};

export default ResultsPage;