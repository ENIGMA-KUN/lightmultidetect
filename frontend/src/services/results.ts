import { apiRequest } from './api';

interface ResultQuery {
    task_id?: string;
    result_id?: string;
}

interface ResultStatus {
    status: string;
    progress: number;
    message?: string;
    result_id?: string;
}

// Query result status by task ID or result ID
export const queryResult = async (query: ResultQuery): Promise<ResultStatus> => {
    return apiRequest<ResultStatus>('/results/query', {
        method: 'POST',
        body: JSON.stringify(query),
    });
};

// Fetch detailed result by result ID
export const fetchResultDetails = async (resultId: string): Promise<any> => {
    return apiRequest<any>(`/results/detail/${resultId}`);
};