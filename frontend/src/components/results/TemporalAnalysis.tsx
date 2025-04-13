import React, { useRef, useEffect } from 'react';

interface TemporalData {
    timestamps: number[];
    values: number[];
    threshold: number;
}

interface TemporalAnalysisProps {
    data: TemporalData;
    mediaType: 'audio' | 'video';
    className?: string;
}

const TemporalAnalysis: React.FC<TemporalAnalysisProps> = ({
    data,
    mediaType,
    className = '',
}) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        // Canvas dimensions
        const width = canvas.width;
        const height = canvas.height;

        // Clear canvas
        ctx.clearRect(0, 0, width, height);

        // Draw background
        ctx.fillStyle = '#1f2937';
        ctx.fillRect(0, 0, width, height);

        // Draw grid
        ctx.strokeStyle = '#374151';
        ctx.lineWidth = 1;

        // Vertical grid lines
        for (let i = 0; i <= 10; i++) {
            const x = (width / 10) * i;
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, height);
            ctx.stroke();
        }

        // Horizontal grid lines
        for (let i = 0; i <= 5; i++) {
            const y = (height / 5) * i;
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(width, y);
            ctx.stroke();
        }

        // Draw threshold line
        const thresholdY = height - (height * data.threshold);
        ctx.strokeStyle = '#f59e0b';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(0, thresholdY);
        ctx.lineTo(width, thresholdY);
        ctx.stroke();

        // Draw data line
        if (data.timestamps.length > 0) {
            const maxTime = Math.max(...data.timestamps);
            const timeScale = width / maxTime;

            ctx.strokeStyle = '#3b82f6';
            ctx.lineWidth = 3;
            ctx.beginPath();

            for (let i = 0; i < data.timestamps.length; i++) {
                const x = data.timestamps[i] * timeScale;
                const y = height - (height * data.values[i]);

                if (i === 0) {
                    ctx.moveTo(x, y);
                } else {
                    ctx.lineTo(x, y);
                }
            }

            ctx.stroke();

            // Draw data points
            for (let i = 0; i < data.timestamps.length; i++) {
                const x = data.timestamps[i] * timeScale;
                const y = height - (height * data.values[i]);

                // Determine point color based on value
                if (data.values[i] > data.threshold) {
                    ctx.fillStyle = '#ef4444';
                } else {
                    ctx.fillStyle = '#10b981';
                }

                ctx.beginPath();
                ctx.arc(x, y, 5, 0, Math.PI * 2);
                ctx.fill();

                // Draw point border
                ctx.strokeStyle = '#f3f4f6';
                ctx.lineWidth = 1;
                ctx.beginPath();
                ctx.arc(x, y, 5, 0, Math.PI * 2);
                ctx.stroke();
            }
        }

        // Draw labels
        ctx.font = '12px Arial';
        ctx.textAlign = 'left';
        ctx.textBaseline = 'top';

        // Threshold label
        ctx.fillStyle = '#f59e0b';
        ctx.fillText(`Threshold (${data.threshold.toFixed(2)})`, 10, thresholdY - 20);

        // Axis labels
        ctx.fillStyle = '#d1d5db';
        ctx.fillText('Time', width - 40, height - 20);
        ctx.fillText('Score', 10, 10);

    }, [data]);

    return (
        <div className={`bg-gray-800 rounded-lg p-4 border border-gray-700 ${className}`}>
            <h3 className="text-lg font-medium text-white mb-4">
                Temporal Analysis
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
                    This chart shows how the deepfake confidence score changes over time in the {mediaType}.
                    Points above the yellow threshold line indicate potential manipulation.
                </p>
                <div className="mt-2 flex items-center space-x-4">
                    <div className="flex items-center">
                        <div className="w-3 h-3 rounded-full bg-green-500 mr-1"></div>
                        <span className="text-gray-400">Authentic</span>
                    </div>
                    <div className="flex items-center">
                        <div className="w-3 h-3 rounded-full bg-red-500 mr-1"></div>
                        <span className="text-gray-400">Manipulated</span>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default TemporalAnalysis;