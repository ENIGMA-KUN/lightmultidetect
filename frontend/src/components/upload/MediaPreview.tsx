import React, { useState, useEffect } from 'react';

interface MediaPreviewProps {
    file: File;
    className?: string;
}

const MediaPreview: React.FC<MediaPreviewProps> = ({ file, className = '' }) => {
    const [preview, setPreview] = useState<string | null>(null);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        if (!file) {
            setPreview(null);
            return;
        }

        const objectUrl = URL.createObjectURL(file);
        setPreview(objectUrl);

        // Clean up
        return () => {
            URL.revokeObjectURL(objectUrl);
        };
    }, [file]);

    if (!preview) {
        return <div className="text-gray-400">Loading preview...</div>;
    }

    // Determine media type based on file type
    const isImage = file.type.startsWith('image/');
    const isAudio = file.type.startsWith('audio/');
    const isVideo = file.type.startsWith('video/');

    return (
        <div className={`rounded-lg overflow-hidden bg-gray-800 border border-gray-700 ${className}`}>
            <div className="bg-gradient-to-r from-gray-900 to-gray-800 text-white p-3 border-b border-gray-700">
                <div className="flex justify-between items-center">
                    <div className="font-medium truncate">{file.name}</div>
                    <div className="text-xs text-gray-400">{(file.size / (1024 * 1024)).toFixed(2)} MB</div>
                </div>
            </div>

            <div className="p-4 bg-black bg-opacity-50">
                {isImage && (
                    <img
                        src={preview}
                        alt="Preview"
                        className="mx-auto max-h-96 object-contain"
                        onError={() => setError("Failed to load image preview")}
                    />
                )}

                {isAudio && (
                    <div className="flex flex-col items-center">
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-24 w-24 text-blue-500 mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3" />
                        </svg>
                        <audio
                            controls
                            src={preview}
                            className="w-full"
                            onError={() => setError("Failed to load audio preview")}
                        />
                    </div>
                )}

                {isVideo && (
                    <video
                        controls
                        src={preview}
                        className="mx-auto max-h-96 w-full"
                        onError={() => setError("Failed to load video preview")}
                    />
                )}

                {error && (
                    <div className="text-red-500 text-center mt-4">{error}</div>
                )}
            </div>
        </div>
    );
};

export default MediaPreview;