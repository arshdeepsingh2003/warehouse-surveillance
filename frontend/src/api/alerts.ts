import axios from 'axios';

const api = axios.create({ baseURL: '/api' });

export const getAlerts = () => api.get('/alerts');
export const acknowledgeAlert = (id: string) => api.patch(`/alerts/${id}/acknowledge`);
