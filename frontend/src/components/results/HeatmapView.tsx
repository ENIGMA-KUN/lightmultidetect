import React, { useState } from 'react';

interface Region {
    x: number;
    y: number;
    width: number;
    height: number;
    confidence: number;
    label?: string;
}

interface HeatmapViewProps {
    heatmapUrl: string;
    regions?: Region[];
    className?: string;
}

const HeatmapView: React.FC<HeatmapViewProps> = ({
    heatmapUrl,
    regions = [],
    className = '',
}) => {
    const [selectedRegion, setSelectedRegion] = useState<Region | null>(null);

    return (
        <div className={`bg-gray-800 rounded-lg p-4 border border-gray-700 ${className}`}>
            <h3 className="text-lg font-medium text-white mb-4">Visual Analysis</h3>

            <div className="relative">
                <img
                    src={heatmapUrl}
                    alt="Deepfake detection heatmap"
                    className="w-full rounded-lg"
                />

                {regions.map((region, index) => (
                    <div
                        key={index}
                        className="absolute border-2 border-blue-500 cursor-pointer transition-all duration-200 hover:border-blue-300"
                        style={{
                            left: `${region.x}px`,
                            top: `${region.y}px`,
                            width: `${region.width}px`,
                            height: `${region.height}px`,
                            backgroundColor: selectedRegion === region ? 'rgba(59, 130, 246, 0.3)' : 'transparent'
                        }}
                        onClick={() => setSelectedRegion(region === selectedRegion ? null : region)}
                    />
                ))}
            </div>

            <div className="mt-4 text-sm text-gray-300">
                <p>
                    This visualization highlights potential manipulated regions. Brighter areas indicate higher probability of manipulation.
                </p>

                {selectedRegion && (
                    <div className="mt-2 p-3 bg-blue-900 bg-opacity-20 rounded-lg border border-blue-800">
                        <h4 className="font-medium text-blue-300">Region Details</h4>
                        <div className="mt-1 text-gray-300">
                            <p>Confidence: {Math.round(selectedRegion.confidence * 100)}%</p>
                            {selectedRegion.label && <p>Type: {selectedRegion.label}</p>}
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
};

export default HeatmapView;