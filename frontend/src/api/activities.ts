import axios from 'axios';

const api = axios.create({ baseURL: '/api' });

export const getActivities = (params?: any) => api.get('/activities', { params });
