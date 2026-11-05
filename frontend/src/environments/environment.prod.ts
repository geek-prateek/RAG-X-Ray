export const environment = {
  production: true,
  // In production / Vercel, default to relative path '/api' so Vercel rewrites or reverse proxies route it cleanly.
  // Can be overridden at runtime via window.__ENV__.API_URL or localStorage.getItem('RAG_API_URL').
  apiUrl: '/api'
};
