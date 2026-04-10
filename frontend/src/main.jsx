import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import { Toaster } from 'react-hot-toast'
import App from './App'
import './index.css'

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <BrowserRouter>
      <App />
      <Toaster position="bottom-right" toastOptions={{
        style: { background:'#12121a', color:'#e2e8f0', border:'1px solid #1e1e2e', borderRadius:'10px' }
      }} />
    </BrowserRouter>
  </React.StrictMode>
)
