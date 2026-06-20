"""Production WSGI entrypoint (waitress) for the laptop-as-server setup.

Launched by pm2 (see ../ecosystem.config.js). We use waitress because gunicorn
does not run on Windows.

trusted_proxy + trusted_proxy_headers make waitress HONOR the X-Forwarded-Proto
header that nginx sets (https for the public Cloudflare host). Without this,
waitress 3.x strips X-Forwarded-* from "untrusted" proxies, so Django sees the
request as http and builds http:// media URLs — which an https page then blocks
as mixed content.
"""
import os

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "agevision_backend.settings")

from waitress import serve
from agevision_backend.wsgi import application

if __name__ == "__main__":
    serve(
        application,
        host="127.0.0.1",
        port=8000,
        threads=8,
        # nginx runs locally and is the only thing that talks to waitress.
        trusted_proxy="127.0.0.1",
        trusted_proxy_count=1,
        trusted_proxy_headers={"x-forwarded-proto"},
    )
