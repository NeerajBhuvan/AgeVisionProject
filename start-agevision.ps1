# Ensure the whole AgeVision production stack is up, then report status.
# Safe to run anytime (idempotent). Most pieces auto-start on boot; run this if
# the site isn't reachable after you open the laptop.
#
#   PowerShell:  .\start-agevision.ps1
#
# Backend, tunnel and MongoDB are Windows SERVICES (start on boot, no windows).
# nginx is started here / by the Startup-folder script.

$PUBLIC = 'https://agevision.thinkblooms.in'
$NGINX  = 'C:\Users\neera\nginx\nginx.exe'
$NGINXP = 'C:\Users\neera\nginx\'

function Line($n,$ok,$d){ $m = if($ok){'[ OK ]'}else{'[FAIL]'}; Write-Host ("{0}  {1,-20} {2}" -f $m,$n,$d) }

function Ensure-Service($name,$label){
  $s = Get-Service $name -ErrorAction SilentlyContinue
  if (-not $s) { Line $label $false 'not installed'; return }
  if ($s.Status -ne 'Running') { try { Start-Service $name -ErrorAction Stop } catch {} ; $s = Get-Service $name }
  Line $label ($s.Status -eq 'Running') $s.Status
}

Write-Host "`n=== Ensuring AgeVision stack ===" -ForegroundColor Cyan
Ensure-Service 'MongoDB'          'MongoDB'
Ensure-Service 'Cloudflared'      'Cloudflare tunnel'
Ensure-Service 'AgeVisionBackend' 'Backend (service)'

# nginx (not a service — start if not running)
if (-not (Get-Process nginx -ErrorAction SilentlyContinue)) {
  Start-Process -FilePath $NGINX -ArgumentList '-p',$NGINXP -WorkingDirectory $NGINXP
  Start-Sleep 2
}
Line 'nginx' ([bool](Get-Process nginx -ErrorAction SilentlyContinue)) ("{0} process(es)" -f (Get-Process nginx -ErrorAction SilentlyContinue | Measure-Object).Count)

Start-Sleep 2
$localOk = $false; try { $localOk = (Invoke-WebRequest 'http://localhost/api/health/' -UseBasicParsing -TimeoutSec 8).StatusCode -eq 200 } catch {}
Line 'Local  (localhost)' $localOk 'http://localhost'
$pubOk = $false; try { $pubOk = (Invoke-WebRequest "$PUBLIC/api/health/" -UseBasicParsing -TimeoutSec 20).StatusCode -eq 200 } catch {}
Line 'Public (tunnel)' $pubOk $PUBLIC

Write-Host ""
if ($localOk -and $pubOk) { Write-Host "All good - open $PUBLIC" -ForegroundColor Green }
else {
  Write-Host "Something is down (see [FAIL] above)." -ForegroundColor Yellow
  Write-Host "  Backend logs:  D:\AU\Project\AgeVisionProject\service\agevision-backend.out.log" -ForegroundColor DarkYellow
  Write-Host "  If a SERVICE shows [FAIL], re-run this from an Administrator PowerShell." -ForegroundColor DarkYellow
}
Write-Host "(First prediction after a restart takes ~25-30s while the model loads.)`n"
