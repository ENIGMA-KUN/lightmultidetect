import React, { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import ResultsPage from '../components/results/ResultsPage';
import Button from '../components/common/Button';
import { useResults } from '../hooks/useResults';

const ResultPage: React.FC = () => {
    const { resultId } = useParams<{ resultId: string }>();
    const navigate = useNavigate();
    const { result, loading, error, fetchResult } = useResults();

    useEffect(() => {
        if (resultId) {
            fetchResult(resultId);
        }
    }, [resultId, fetchResult]);

    const handleNewAnalysis = () => {
        navigate('/detect');
    };

    if (loading) {
        return (
            <div className="flex justify-center items-center min-h-[400px]">
                <div className="text-center">
                    <div className="inline-block animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500 mb-4"></div>
                    <p className="text-gray-300">Loading result data...</p>
                </div>
            </div>
        );
    }

    if (error || !result) {
        return (
            <div className="text-center py-12">
                <svg className="h-16 w-16 text-red-500 mx-auto mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <h2 className="text-2xl font-bold text-white mb-2">Result Not Found</h2>
                <p className="text-gray-400 mb-6">
                    {error || "We couldn't find the result you're looking for. It may have expired or been removed."}
                </p>
                <Button variant="primary" onClick={handleNewAnalysis}>
                    Start New Analysis
                </Button>
            </div>
        );
    }

    return (
        <div className="max-w-6xl mx-auto">
            <h1 className="text-3xl font-bold mb-6">Analysis Results</h1>

            <ResultsPage
                result={result}
                mediaType={result.media_type as 'image' | 'audio' | 'video'}
                onNewAnalysis={handleNewAnalysis}
            />
        </div>
    );
};

export default ResultPage;