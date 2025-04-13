import React, { useRef, useEffect } from 'react';

interface FrequencyAnalysisProps {
    data: any; // Replace with proper type
    className?: string;
}

const FrequencyAnalysis: React.FC<FrequencyAnalysisProps> = ({
    data,
    className = '',
}) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        // Simplified visualization for frequency data
        // In a real application, you would use more sophisticated visualization

        // Canvas dimensions
        const width = canvas.width;
        const height = canvas.height;

        // Clear canvas
        ctx.clearRect(0, 0, width, height);

        // Draw background
        ctx.fillStyle = '#1f2937';
        ctx.fillRect(0, 0, width, height);

        // Get data to visualize
        let frequencyData: [string, number][] = [];

        if (data.low_freq_energy !== undefined) {
            // If we have specific frequency bands
            frequencyData = [
                ['Low', data.low_freq_energy],
                ['Mid', data.mid_freq_energy],
                ['High', data.high_freq_energy],
            ];
        } else {
            // Generic case - just use whatever data we have
            frequencyData = Object.entries(data)
                .filter(([key, value]) => typeof value === 'number')
                .map(([key, value]): [string, number] => [key, value as number]);
        }

        // Find max value for scaling
        const maxValue = Math.max(...frequencyData.map(([_, value]) => value));

        // Bar width
        const barWidth = width / frequencyData.length;
        const padding = barWidth * 0.15;

        // Draw bars
        frequencyData.forEach(([label, value], index) => {
            const barHeight = (value / maxValue) * (height - 60);
            const x = index * barWidth + padding;
            const y = height - barHeight - 30;
            const barWidthWithPadding = barWidth - (padding * 2);

            // Create gradient
            const gradient = ctx.createLinearGradient(x, y, x, height - 30);
            gradient.addColorStop(0, '#3b82f6');
            gradient.addColorStop(1, '#1d4ed8');

            // Draw bar
            ctx.fillStyle = gradient;
            ctx.fillRect(x, y, barWidthWithPadding, barHeight);

            // Draw bar border
            ctx.strokeStyle = '#60a5fa';
            ctx.lineWidth = 1;
            ctx.strokeRect(x, y, barWidthWithPadding, barHeight);

            // Draw label
            ctx.fillStyle = '#d1d5db';
            ctx.font = '12px Arial';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'top';
            ctx.fillText(label, x + (barWidthWithPadding / 2), height - 20);

            // Draw value
            ctx.fillStyle = '#9ca3af';
            ctx.font = '10px Arial';
            ctx.fillText(value.toFixed(2), x + (barWidthWithPadding / 2), y - 15);
        });

    }, [data]);

    return (
        <div className={`bg-gray-800 rounded-lg p-4 border border-gray-700 ${className}`}>
            <h3 className="text-lg font-medium text-white mb-4">
                Frequency Analysis
            </h3>

            <div className="relative">
                <canvas
                    ref={canvasRef}
                    width={500}
                    height={300}
                    className="w-full rounded-lg"
                />
            </div>

            <div className="mt-4 text-sm text-gray-300">
                <p>
                    This chart shows frequency domain analysis, which helps detect GAN-generated content.
                    Manipulated content often shows abnormal patterns in high frequency components.
                </p>

                {data.high_to_low_ratio !== undefined && (
                    <div className="mt-2 p-3 bg-gray-900 rounded-lg">
                        <div className="flex justify-between items-center">
                            <span className="text-gray-400">High/Low Ratio:</span>
                            <span className={`font-medium ${data.high_to_low_ratio > 0.5 ? 'text-red-400' : 'text-green-400'
                                }`}>
                                {data.high_to_low_ratio.toFixed(3)}
                            </span>
                        </div>
                        <p className="text-xs text-gray-500 mt-1">
                            Higher values may indicate GAN-generated content
                        </p>
                    </div>
                )}
            </div>
        </div>
    );
};

export default FrequencyAnalysis;