import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import UploadArea from '../components/upload/UploadArea';
import MediaPreview from '../components/upload/MediaPreview';
import UploadOptions from '../components/upload/UploadOptions';
import AnalysisProgress from '../components/analysis/AnalysisProgress';
import { useDetection } from '../hooks/useDetection';

const DetectionPage: React.FC = () => {
    const navigate = useNavigate();
    const [activeTab, setActiveTab] = useState<'image' | 'audio' | 'video'>('image');
    const [selectedFile, setSelectedFile] = useState<File | null>(null);
    const [options, setOptions] = useState({
        detailedAnalysis: true,
        confidenceThreshold: 0.5,
    });

    const {
        isProcessing,
        progress,
        statusMessage,
        resultId,
        startDetection,
        checkDetectionStatus,
    } = useDetection();

    useEffect(() => {
        if (resultId) {
            navigate(`/result/${resultId}`);
        }
    }, [resultId, navigate]);

    const handleFileSelected = (file: File) => {
        setSelectedFile(file);
    };

    const handleOptionChange = (name: string, value: any) => {
        setOptions(prev => ({
            ...prev,
            [name]: value,
        }));
    };

    const handleSubmit = async () => {
        if (!selectedFile) return;

        await startDetection(selectedFile, activeTab, options.detailedAnalysis, options.confidenceThreshold);
    };

    const handleCancel = () => {
        setSelectedFile(null);
    };

    // Get accepted file types based on active tab
    const getAcceptedFileTypes = () => {
        switch (activeTab) {
            case 'image':
                return ['image/jpeg', 'image/png'];
            case 'audio':
                return ['audio/mpeg', 'audio/wav', 'audio/x-wav'];
            case 'video':
                return ['video/mp4', 'video/quicktime', 'video/x-msvideo'];
            default:
                return [];
        }
    };

    return (
        <div className="max-w-6xl mx-auto">
            <h1 className="text-3xl font-bold mb-6">Deepfake Detection</h1>

            {isProcessing ? (
                <AnalysisProgress
                    progress={progress}
                    message={statusMessage}
                    mediaType={activeTab}
                />
            ) : (
                <div className="bg-gray-800 rounded-lg border border-gray-700 overflow-hidden">
                    {/* Tabs */}
                    <div className="bg-gray-900 p-1 flex">
                        <button
                            onClick={() => {
                                setActiveTab('image');
                                setSelectedFile(null);
                            }}
                            className={`
                flex-1 py-2 rounded-md transition-colors text-center
                ${activeTab === 'image'
                                    ? 'bg-gray-800 text-white'
                                    : 'text-gray-400 hover:text-white'}
              `}
                        >
                            Image
                        </button>

                        <button
                            onClick={() => {
                                setActiveTab('audio');
                                setSelectedFile(null);
                            }}
                            className={`
                flex-1 py-2 rounded-md transition-colors text-center
                ${activeTab === 'audio'
                                    ? 'bg-gray-800 text-white'
                                    : 'text-gray-400 hover:text-white'}
              `}
                        >
                            Audio
                        </button>

                        <button
                            onClick={() => {
                                setActiveTab('video');
                                setSelectedFile(null);
                            }}
                            className={`
                flex-1 py-2 rounded-md transition-colors text-center
                ${activeTab === 'video'
                                    ? 'bg-gray-800 text-white'
                                    : 'text-gray-400 hover:text-white'}
              `}
                        >
                            Video
                        </button>
                    </div>

                    {/* Content */}
                    <div className="p-6">
                        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                            <div className="lg:col-span-2">
                                {!selectedFile ? (
                                    <UploadArea
                                        onFileSelected={handleFileSelected}
                                        acceptedFileTypes={getAcceptedFileTypes()}
                                        mediaType={activeTab}
                                    />
                                ) : (
                                    <MediaPreview
                                        file={selectedFile}
                                    />
                                )}
                            </div>

                            <div className="lg:col-span-1">
                                {selectedFile && (
                                    <UploadOptions
                                        onSubmit={handleSubmit}
                                        onCancel={handleCancel}
                                        onOptionChange={handleOptionChange}
                                        options={options}
                                        isProcessing={isProcessing}
                                    />
                                )}
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

export default DetectionPage;