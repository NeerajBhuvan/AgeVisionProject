# Starts the AgeVision Django backend for the local demo.
#   PowerShell:  .\run-backend.ps1
# Runs migrations (creates/updates the local SQLite auth DB) then serves on
# 0.0.0.0:8000 so both localhost and LAN devices can reach it.

$ErrorActionPreference = 'Stop'
$venvPy = 'D:\AU\Project\agevision_env\Scripts\python.exe'
$backend = Join-Path $PSScriptRoot 'agevision_backend'

if (-not (Test-Path $venvPy)) {
    Write-Error "venv python not found at $venvPy — activate your env or fix the path."
    exit 1
}

Write-Host '==> Applying migrations (auth/sessions/JWT blacklist)...' -ForegroundColor Cyan
& $venvPy (Join-Path $backend 'manage.py') migrate --noinput

Write-Host '==> Starting Django on http://0.0.0.0:8000 ...' -ForegroundColor Cyan
& $venvPy (Join-Path $backend 'manage.py') runserver 0.0.0.0:8000
