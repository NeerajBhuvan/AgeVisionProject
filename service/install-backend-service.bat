@echo off
REM Run this ONCE as Administrator (right-click -> Run as administrator).
REM Installs the AgeVision backend as a Windows service (no console window,
REM starts on boot). Safe to re-run (reinstalls).
cd /d "%~dp0"

echo Removing any previous service...
agevision-backend.exe stop 2>nul
agevision-backend.exe uninstall 2>nul
timeout /t 2 /nobreak >nul

echo Installing service...
agevision-backend.exe install
agevision-backend.exe start

echo.
echo Current status:
sc query AgeVisionBackend | findstr STATE
echo.
echo Done. You can close this window.
pause
