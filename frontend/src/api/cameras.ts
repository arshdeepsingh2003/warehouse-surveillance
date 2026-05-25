import axios from 'axios';

const api = axios.create({ baseURL: '/api' });

export const getCameras = () => api.get('/cameras');
export const getCamera = (id: string) => api.get(`/cameras/${id}`);
