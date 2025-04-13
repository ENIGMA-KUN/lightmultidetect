import React, { useEffect, useRef } from 'react';

interface ConfidenceGaugeProps {
    score: number;
    size?: 'sm' | 'md' | 'lg';
    showLabel?: boolean;
    className?: string;
}

const ConfidenceGauge: React.FC<ConfidenceGaugeProps> = ({
    score,
    size = 'md',
    showLabel = true,
    className = '',
}) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);

    // Size dimensions
    const sizeDimensions = {
        sm: { width: 120, height: 60, fontSize: 12, thickness: 8 },
        md: { width: 180, height: 90, fontSize: 16, thickness: 12 },
        lg: { width: 240, height: 120, fontSize: 20, thickness: 16 },
    };

    const { width, height, fontSize, thickness } = sizeDimensions[size];

    // Cap score between 0 and 1
    const safeScore = Math.min(1, Math.max(0, score));

    // Determine color based on score
    const getColor = (value: number): string => {
        // Green to red gradient
        if (value < 0.5) {
            // Green to yellow (0 to 0.5)
            const r = Math.floor(255 * (value * 2));
            const g = 255;
            return `rgb(${r}, ${g}, 0)`;
        } else {
            // Yellow to red (0.5 to 1)
            const g = Math.floor(255 * (1 - (value - 0.5) * 2));
            return `rgb(255, ${g}, 0)`;
        }
    };

    // Get label based on score
    const getLabel = (value: number): string => {
        if (value < 0.3) return "Likely Real";
        if (value < 0.5) return "Possibly Real";
        if (value < 0.7) return "Possibly Fake";
        return "Likely Fake";
    };

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        // Clear canvas
        ctx.clearRect(0, 0, width, height);

        // Set up dimensions
        const centerX = width / 2;
        const centerY = height * 0.8;
        const radius = Math.min(width, height) * 0.8;

        // Draw background arc
        ctx.beginPath();
        ctx.arc(centerX, centerY, radius / 2, Math.PI, 0, false);
        ctx.lineWidth = thickness;
        ctx.strokeStyle = '#333';
        ctx.stroke();

        // Draw value arc
        const angle = Math.PI * (1 - safeScore);
        ctx.beginPath();
        ctx.arc(centerX, centerY, radius / 2, Math.PI, angle, false);
        ctx.lineWidth = thickness;
        ctx.strokeStyle = getColor(safeScore);
        ctx.stroke();

        // Draw score text
        ctx.font = `bold ${fontSize}px Arial`;
        ctx.fillStyle = getColor(safeScore);
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(`${Math.round(safeScore * 100)}%`, centerX, centerY - 15);

        // Draw label text
        if (showLabel) {
            ctx.font = `${fontSize * 0.8}px Arial`;
            ctx.fillStyle = '#fff';
            ctx.fillText(getLabel(safeScore), centerX, centerY + 15);
        }

        // Draw min/max labels
        ctx.font = `${fontSize * 0.7}px Arial`;
        ctx.fillStyle = '#aaa';
        ctx.textAlign = 'left';
        ctx.fillText('Real', 10, centerY);
        ctx.textAlign = 'right';
        ctx.fillText('Fake', width - 10, centerY);

    }, [safeScore, width, height, fontSize, thickness, showLabel]);

    return (
        <div className={`flex flex-col items-center ${className}`}>
            <canvas
                ref={canvasRef}
                width={width}
                height={height}
                className="gauge-canvas"
            />
        </div>
    );
};

export default ConfidenceGauge;