// Prefer env var; otherwise default to Render backend in production builds
export const API_BASE =
    process.env.REACT_APP_API_BASE ||
    (typeof window !== 'undefined' ? 'https://spam-detection-api-bdqx.onrender.com' : '/api');

