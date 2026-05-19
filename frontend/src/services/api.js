import axios from 'axios'

const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const api = axios.create({ baseURL: `${BASE_URL}/api`, timeout: 120000 })
api.interceptors.response.use(r => r.data, err => Promise.reject(err))

export const fetchSummary    = ()  => api.get('/mlflow/summary')
export const fetchRuns       = ()  => api.get('/mlflow/runs')
export const fetchComparison = ()  => api.get('/mlflow/comparison')
export const fetchModels     = ()  => api.get('/train/models')
export const trainModel      = (p) => api.post('/train', p)
export const fetchPreview    = ()  => api.get('/data/preview')
export const uploadDataset   = (f) => api.post('/data/upload', f, { headers: { 'Content-Type': 'multipart/form-data' } })
export const predict         = (p) => api.post('/predict', p)
