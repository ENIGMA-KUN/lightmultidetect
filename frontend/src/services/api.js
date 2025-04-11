import axios from 'axios';

const axiosInstance = axios.create({
  baseURL: process.env.REACT_APP_API_URL || 'http://localhost:8000',
  headers: {
    'Content-Type': 'application/json'
  }
});

const api = {
  setAuthToken: (token) => {
    if (token) {
      axiosInstance.defaults.headers.common['Authorization'] = `Bearer ${token}`;
    } else {
      delete axiosInstance.defaults.headers.common['Authorization'];
    }
  },

  // Auth endpoints
  login: (credentials) => axiosInstance.post('/auth/login', credentials),
  register: (userData) => axiosInstance.post('/auth/register', userData),
  getProfile: () => axiosInstance.get('/auth/me'),
  updateProfile: (data) => axiosInstance.put('/auth/me', data),
  changePassword: (data) => axiosInstance.put('/auth/change-password', data),

  // Analysis endpoints
  uploadFiles: (formData) => axiosInstance.post('/analysis/upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data'
    }
  }),
  getAnalysisResults: (params) => axiosInstance.get('/analysis/results', { params }),
  getAnalysisDetail: (id) => axiosInstance.get(`/analysis/results/${id}`),
  deleteAnalysis: (id) => axiosInstance.delete(`/analysis/results/${id}`),

  // Dashboard endpoints
  getDashboardStats: () => axiosInstance.get('/dashboard/stats'),
  getRecentAnalyses: () => axiosInstance.get('/dashboard/recent-analyses'),
  getAnalysisHistory: (params) => axiosInstance.get('/dashboard/analysis-history', { params })
};

export default api; 