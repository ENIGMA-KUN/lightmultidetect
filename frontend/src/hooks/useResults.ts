import { useState, useCallback } from 'react';
import { fetchResultDetails } from '../services/results';

export const useResults = () => {
    const [result, setResult] = useState<any | null>(null);
    const [loading, setLoading] = useState<boolean>(false);
    const [error, setError] = useState<string | null>(null);

    // Fetch detailed result
    const fetchResult = useCallback(async (resultId: string) => {
        setLoading(true);
        setError(null);

        try {
            const data = await fetchResultDetails(resultId);
            setResult(data);
        } catch (error) {
            setError(error instanceof Error ? error.message : 'Failed to fetch result');
            setResult(null);
        } finally {
            setLoading(false);
        }
    }, []);

    return {
        result,
        loading,
        error,
        fetchResult,
    };
};