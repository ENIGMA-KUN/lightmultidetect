import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import Button from '../common/Button';

interface UploadAreaProps {
    onFileSelected: (file: File) => void;
    acceptedFileTypes: string[];
    mediaType: 'image' | 'audio' | 'video';
    maxSize?: number;
    className?: string;
}

const UploadArea: React.FC<UploadAreaProps> = ({
    onFileSelected,
    acceptedFileTypes,
    mediaType,
    maxSize = 100 * 1024 * 1024, // 100MB default
    className = '',
}) => {
    const [error, setError] = useState<string | null>(null);

    const onDrop = useCallback((acceptedFiles: File[], rejectedFiles: any[]) => {
        if (rejectedFiles.length > 0) {
            const { code } = rejectedFiles[0].errors[0];
            if (code === 'file-too-large') {
                setError(`File is too large. Max size is ${(maxSize / (1024 * 1024)).toFixed(0)}MB.`);
            } else if (code === 'file-invalid-type') {
                setError(`Invalid file type. Accepted types: ${acceptedFileTypes.join(', ')}`);
            } else {
                setError('Invalid file. Please try again.');
            }
            return;
        }

        if (acceptedFiles.length > 0) {
            setError(null);
            onFileSelected(acceptedFiles[0]);
        }
    }, [onFileSelected, acceptedFileTypes, maxSize]);

    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        onDrop,
        accept: acceptedFileTypes.reduce((acc, type) => ({ ...acc, [type]: [] }), {}),
        maxSize,
        multiple: false,
    });

    // Media type icon
    const getMediaIcon = () => {
        switch (mediaType) {
            case 'image':
                return (
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-12 w-12 text-blue-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                    </svg>
                );
            case 'audio':
                return (
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-12 w-12 text-blue-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3" />
                    </svg>
                );
            case 'video':
                return (
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-12 w-12 text-blue-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                    </svg>
                );
            default:
                return (
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-12 w-12 text-blue-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                    </svg>
                );
        }
    };

    const mediaTypeLabel = mediaType.charAt(0).toUpperCase() + mediaType.slice(1);

    return (
        <div className={`w-full ${className}`}>
            <div
                {...getRootProps()}
                className={`
          border-2 border-dashed rounded-lg p-8 
          transition-all duration-300 ease-in-out
          ${isDragActive ? 'border-blue-500 bg-blue-50 bg-opacity-10' : 'border-gray-600 hover:border-blue-500 hover:bg-blue-50 hover:bg-opacity-5'}
          focus:outline-none focus:border-blue-500
          h-64 flex flex-col items-center justify-center cursor-pointer
        `}
            >
                <input {...getInputProps()} />

                {getMediaIcon()}

                <p className="mt-4 text-lg font-medium text-gray-300">
                    {isDragActive ? (
                        `Drop your ${mediaType} here...`
                    ) : (
                        `Drag & drop your ${mediaType} here, or click to select`
                    )}
                </p>

                <p className="mt-2 text-sm text-gray-400">
                    Supported formats: {acceptedFileTypes.join(', ')}
                </p>

                <Button
                    variant="secondary"
                    size="sm"
                    className="mt-4"
                >
                    Select {mediaTypeLabel}
                </Button>
            </div>

            {error && (
                <div className="mt-2 text-red-500 text-sm">{error}</div>
            )}
        </div>
    );
};

export default UploadArea;