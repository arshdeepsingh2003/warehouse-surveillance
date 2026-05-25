import axios from 'axios';

const api = axios.create({ baseURL: '/api' });

export const getAnalytics = (params?: any) => api.get('/analytics', { params });
