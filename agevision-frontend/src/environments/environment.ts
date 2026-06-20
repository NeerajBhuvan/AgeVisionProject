// Dev build (`ng serve`). API is same-origin via proxy.conf.json, which
// forwards /api and /media to the Django backend on :8000. Relative URLs mean
// the app works whether opened on localhost OR a LAN IP (no hard-coded host).
export const environment = {
  production: false,
  apiUrl: '/api'
};
