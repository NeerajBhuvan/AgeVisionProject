// Production build served by nginx, which reverse-proxies /api to Django on the
// same origin. A relative URL means the same bundle works at http://localhost
// AND at the public https://<your-subdomain> via the Cloudflare tunnel.
export const environment = {
  production: true,
  apiUrl: '/api'
};
