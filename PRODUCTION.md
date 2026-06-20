# AgeVision — laptop-as-server (production setup)

Runs the app in production mode on this laptop. **LIVE publicly at
https://agevision.thinkblooms.in** (via Cloudflare Tunnel) and locally at
http://localhost — reachable anywhere the laptop is on.

```
  Anywhere ──HTTPS──▶ https://agevision.thinkblooms.in
                         │  (Cloudflare edge: TLS, hides home IP)
                         │  encrypted tunnel (outbound; no port-forwarding)
                   cloudflared  ◀── Windows service "Cloudflared"
                         │
  Browser ──▶ nginx :80 ─┬─ /        Angular production build (static, gzipped)
                         ├─ /api/*   ─▶ waitress :8000  (Django, DEBUG=False)
                         ├─ /media/* ─▶ waitress :8000      ▲
                         └─ /static/*─▶ waitress :8000      └─ Windows service "AgeVisionBackend"
                                            │
                       MiVOLO (local CPU) + Modal FADING + MongoDB + SQLite
```

## Components (3 of 4 are Windows services → start on boot, no windows)
| Piece | How it runs | Boot |
|---|---|---|
| **Backend** (Django via waitress / `serve.py`) | Windows service **AgeVisionBackend** (WinSW, in `service/`) | Automatic |
| **Cloudflare tunnel** | Windows service **Cloudflared** | Automatic |
| **MongoDB** | Windows service **MongoDB** | Automatic |
| **nginx** (`C:\Users\neera\nginx`) | started by `agevision-nginx.bat` in the Startup folder | on logon |

> The backend used to run under pm2 but that popped a recurring console window;
> it's now a service (`service/agevision-backend.xml`). pm2 is no longer used
> ([ecosystem.config.js](ecosystem.config.js) is an empty stub).

## Opening the laptop → getting the site up
Normally everything auto-starts. If the site isn't reachable, double-click
**"Start AgeVision"** on the Desktop (runs [start-agevision.ps1](start-agevision.ps1)),
which ensures all 4 pieces and prints a status table.

## Day-to-day management
```powershell
# Backend service
sc query AgeVisionBackend                 # status
Restart-Service AgeVisionBackend           # restart (admin) — after backend code changes
Get-Content service\agevision-backend.out.log -Tail 30   # logs

# Tunnel / Mongo services
Get-Service Cloudflared, MongoDB
Restart-Service Cloudflared                # (admin)

# nginx
C:\Users\neera\nginx\nginx.exe -p C:\Users\neera\nginx\ -s reload   # after editing nginx.conf
C:\Users\neera\nginx\nginx.exe -p C:\Users\neera\nginx\             # start
```

## After changing code
- **Frontend:** `cd agevision-frontend; npx ng build --configuration=production` (nginx serves the new files immediately).
- **Backend:** `Restart-Service AgeVisionBackend` (admin). If static changed, run `collectstatic` first.

## ⚠️ Do NOT run these (they pop a window and clash on port 8000)
`python manage.py runserver`, `run-backend.ps1`, or `pm2 start ...`. The service
already runs the API 24/7 with no window. To control it, use `sc` / `Restart-Service`.

## Config locations
- backend service definition + logs: [service/agevision-backend.xml](service/agevision-backend.xml), `service/*.log`
- waitress launcher (trusted_proxy for https): [agevision_backend/serve.py](agevision_backend/serve.py)
- secrets (SECRET_KEY, MONGO_URI, FADING_MODAL_ENDPOINT): `agevision_backend/.env` (git-ignored)
- production env (DEBUG, ALLOWED_HOSTS, CORS, CSRF, BEHIND_PROXY): set in the service XML
- nginx config (tracked copy): [nginx/nginx.conf](nginx/nginx.conf) → copy to `C:\Users\neera\nginx\conf\nginx.conf`

## Public access notes
- cloudflared installed as the Windows service "Cloudflared" (token-based; the
  hostname → `http://localhost:80` mapping lives in the friend's Cloudflare dashboard).
- The hostname is in the service XML's `ALLOWED_HOSTS`/`CORS_ORIGINS`/`CSRF_TRUSTED_ORIGINS`
  and nginx `server_name`. nginx forces `X-Forwarded-Proto: https` for this host and
  serve.py's `trusted_proxy` honors it → correct `https://` media URLs.
- **To change the public hostname:** update the service XML (reinstall via
  `service/install-backend-service.bat` as admin), nginx `server_name` + the `map $host`
  in nginx.conf (reload), and have the friend update the dashboard mapping.

## Keep it online
- Disable sleep (Power & sleep → "When plugged in, turn off after: Never").
- Needs internet (tunnel + Modal FADING for progression).
- First prediction after a restart takes ~25–30s (model load), then ~2s each.
