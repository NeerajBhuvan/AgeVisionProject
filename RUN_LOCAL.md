# Running AgeVision locally

Everything runs on this laptop. The **only** cloud dependency left is the
age-progression FADING model, which runs on a Modal GPU because diffusion is
impractical on a CPU. **Age prediction now runs fully locally on the CPU** —
no cloud, no cold start, no hanging requests.

```
  Browser ─▶ ng serve :4200 ─┬─ /        Angular SPA
        (proxy.conf.json)     ├─ /api/*  ─▶ Django :8000 ─┬─ MongoDB (local service)
                              └─ /media/* ─▶ Django :8000  ├─ SQLite (local file)
                                                           ├─ MiVOLO/YOLO/emotion (local CPU)
                                                           └─ Modal GPU (FADING progression only)
```

Already on this machine: the `agevision_env` venv (with torch+cpu, ultralytics,
transformers, deepface), Node 20, and a running local **MongoDB** service.

## Run it (two terminals)

**Terminal 1 — backend:**
```powershell
.\run-backend.ps1
```
(applies migrations, then serves on `http://0.0.0.0:8000`)

**Terminal 2 — frontend:**
```powershell
cd agevision-frontend
ng serve                       # localhost demo
# ng serve --host 0.0.0.0      # also reachable from other devices on the LAN
```

Open **http://localhost:4200**.

## First-time login
Register through the app's sign-up page, or create an admin user:
```powershell
D:\AU\Project\agevision_env\Scripts\python.exe agevision_backend\manage.py createsuperuser
```
Users live in `agevision_backend/db.sqlite3` and persist across restarts.

## First prediction is slow (one-time)
The first `/api/predict/` after a server start loads the MiVOLO + YOLO + emotion
weights into memory (~10–30s; weights are cached on disk after the first ever
run). Every prediction after that returns in a few seconds. Run one prediction
right after starting the server so the live demo is instant.

## Age progression (FADING) needs the Modal endpoint
Progression calls the Modal FADING GPU. Its URL is in `agevision_backend/.env`
(`FADING_MODAL_ENDPOINT`). The first call after idle cold-starts (~60–180s);
warm it before presenting. SAM / Fast-Aging fall back to the OpenCV pipeline
locally since there's no local GPU.

## Notes
- **Config** lives in `agevision_backend/.env` (git-ignored): local Mongo +
  the FADING Modal endpoint.
- **Mongo not running?** `Start-Service MongoDB` (PowerShell as admin).
- All Cloud Run / Firebase / GCS / Docker deployment files were removed — this
  is a local-only project now.
