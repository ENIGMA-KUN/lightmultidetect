import React from 'react';

interface ExplanationPanelProps {
    result: any; // Replace with proper type
    mediaType: 'image' | 'audio' | 'video';
    className?: string;
}

const ExplanationPanel: React.FC<ExplanationPanelProps> = ({
    result,
    mediaType,
    className = '',
}) => {
    // Get explanation based on media type and result
    const getExplanation = () => {
        if (result.is_fake) {
            switch (mediaType) {
                case 'image':
                    return (
                        <>
                            <p>
                                Our analysis detected potential manipulation in this image with {Math.round(result.confidence_score * 100)}% confidence.
                                The following aspects of the image raised concerns:
                            </p>
                            <ul className="list-disc pl-5 mt-2 space-y-1">
                                {result.confidence_score > 0.7 && (
                                    <li>Strong visual artifacts associated with GAN-generated content</li>
                                )}
                                {result.detection_details?.frequency_analysis?.high_to_low_ratio > 0.5 && (
                                    <li>Abnormal frequency patterns typically seen in synthetic images</li>
                                )}
                                {result.detection_details?.artifact_detection?.texture_inconsistencies > 0.5 && (
                                    <li>Inconsistent texture patterns across the image</li>
                                )}
                                <li>Other detection patterns matching known deepfake signatures</li>
                            </ul>
                        </>
                    );
                case 'audio':
                    return (
                        <>
                            <p>
                                Our analysis detected potential manipulation in this audio with {Math.round(result.confidence_score * 100)}% confidence.
                                The following aspects of the audio raised concerns:
                            </p>
                            <ul className="list-disc pl-5 mt-2 space-y-1">
                                {result.detection_details?.voice_characteristics?.formant_analysis > 0.5 && (
                                    <li>Unnatural voice formant patterns typical of synthetic speech</li>
                                )}
                                {result.confidence_score > 0.7 && (
                                    <li>Algorithmic artifacts in the audio spectrum</li>
                                )}
                                {result.detection_details?.temporal_analysis?.consistency_score < 0.4 && (
                                    <li>Temporal inconsistencies throughout the audio</li>
                                )}
                                <li>Other detection patterns matching known deepfake signatures</li>
                            </ul>
                        </>
                    );
                case 'video':
                    return (
                        <>
                            <p>
                                Our analysis detected potential manipulation in this video with {Math.round(result.confidence_score * 100)}% confidence.
                                The following aspects of the video raised concerns:
                            </p>
                            <ul className="list-disc pl-5 mt-2 space-y-1">
                                {result.detection_details?.temporal_inconsistency > 0.5 && (
                                    <li>Temporal inconsistencies between video frames</li>
                                )}
                                {result.detection_details?.lip_sync_analysis?.lip_sync_score < 0.5 && (
                                    <li>Lip synchronization issues between speech and mouth movements</li>
                                )}
                                {result.confidence_score > 0.7 && (
                                    <li>Visual artifacts in facial features typical of face swapping</li>
                                )}
                                <li>Other detection patterns matching known deepfake signatures</li>
                            </ul>
                        </>
                    );
                default:
                    return <p>Analysis details not available for this media type.</p>;
            }
        } else {
            // Content appears authentic
            return (
                <>
                    <p>
                        Our analysis indicates this {mediaType} is likely authentic with {Math.round((1 - result.confidence_score) * 100)}% confidence.
                        We did not detect significant manipulation indicators that are typically present in deepfakes:
                    </p>
                    <ul className="list-disc pl-5 mt-2 space-y-1">
                        <li>No significant visual artifacts associated with manipulation</li>
                        <li>Natural consistency in {mediaType === 'image' ? 'visual elements' : mediaType === 'audio' ? 'audio patterns' : 'temporal elements'}</li>
                        <li>No unusual patterns in frequency analysis</li>
                        <li>Overall characteristics consistent with authentic {mediaType}s</li>
                    </ul>
                </>
            );
        }
    };

    // Get potential limitations based on confidence
    const getLimitations = () => {
        const isBorderlineConfidence = result.confidence_score > 0.3 && result.confidence_score < 0.7;

        return (
            <>
                <h4 className="font-medium text-gray-300 mb-1">Limitations to consider:</h4>
                <ul className="list-disc pl-5 space-y-1 text-gray-400">
                    {isBorderlineConfidence && (
                        <li>This result has borderline confidence and should be interpreted with caution</li>
                    )}
                    <li>Our detection technology may not identify all types of manipulations</li>
                    <li>New deepfake techniques are constantly evolving</li>
                    <li>Image/video quality can affect detection accuracy</li>
                </ul>
            </>
        );
    };

    return (
        <div className={`bg-gray-800 rounded-lg p-4 border border-gray-700 ${className}`}>
            <h3 className="text-lg font-medium text-white mb-4">Analysis Explanation</h3>

            <div className="prose prose-invert max-w-none text-gray-300">
                {getExplanation()}

                <div className="mt-4 p-3 bg-gray-900 bg-opacity-50 rounded-lg text-sm">
                    {getLimitations()}
                </div>
            </div>
        </div>
    );
};

export default ExplanationPanel;