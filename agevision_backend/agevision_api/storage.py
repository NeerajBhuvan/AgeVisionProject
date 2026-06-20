"""
Media storage helper (local disk).

Writes uploaded + generated images under MEDIA_ROOT and returns URLs built
from the request host, so they work on localhost and across the LAN. Django
serves them via the /media/ route (see agevision_backend/urls.py).

(Cloud/GCS storage was removed — this project now runs entirely on the local
machine.)
"""

import logging
import os
import shutil
from typing import Optional, Union

from django.conf import settings

logger = logging.getLogger(__name__)


def save_media(source: Union[bytes, "object"], relative_path: str) -> str:
    """
    Persist `source` at `relative_path` (e.g. "predictions/<uuid>.jpg").

    `source` may be raw bytes, a Django UploadedFile (anything with .chunks()),
    or a file-like with .read(). Returns the relative `/media/...` URL; callers
    pass `request` to media_url() (or build_absolute_uri) for an absolute URL.
    """
    dest = os.path.join(settings.MEDIA_ROOT, relative_path)
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    with open(dest, 'wb') as out:
        if isinstance(source, (bytes, bytearray)):
            out.write(source)
        elif hasattr(source, 'chunks'):
            for chunk in source.chunks():
                out.write(chunk)
        elif hasattr(source, 'read'):
            data = source.read()
            out.write(data if isinstance(data, (bytes, bytearray))
                      else data.encode('utf-8'))
        else:
            raise TypeError(f"Unsupported source type for save_media: {type(source)!r}")

    return f"{settings.MEDIA_URL}{relative_path}"


def copy_to_media(local_path: str, relative_path: str) -> str:
    """Copy a file already on disk (e.g. a GAN output) into media storage."""
    dest = os.path.join(settings.MEDIA_ROOT, relative_path)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.copy2(local_path, dest)
    return f"{settings.MEDIA_URL}{relative_path}"


def media_url(relative_path: str, request=None) -> Optional[str]:
    """
    Build the URL for an already-stored relative path. Pass `request` so the
    URL comes back absolute (host:port included), which the frontend needs when
    the API and the SPA are on different origins.
    """
    if not relative_path:
        return None

    rel = f"{settings.MEDIA_URL}{relative_path}"
    if request is not None:
        return request.build_absolute_uri(rel)
    return rel
