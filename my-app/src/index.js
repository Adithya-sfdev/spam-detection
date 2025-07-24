import React from 'react'
import ReactDOM from 'react-dom/client'
import { GoogleOAuthProvider } from '@react-oauth/google'
import './index.css'
import App from './App'
import reportWebVitals from './reportWebVitals'

const root = ReactDOM.createRoot(document.getElementById('root'))

/**
 * Wrap your entire app in GoogleOAuthProvider.
 * The clientId should be set in your .env file as REACT_APP_GOOGLE_CLIENT_ID.
 */
root.render(
  <React.StrictMode>
    <GoogleOAuthProvider clientId={process.env.REACT_APP_GOOGLE_CLIENT_ID}>
      <App />
    </GoogleOAuthProvider>
  </React.StrictMode>
)

// Optional: Report web vitals (performance)
reportWebVitals()
