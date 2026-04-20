"""
AgeVision — Upload Checkpoints to Google Drive
================================================
Uploads all ML model checkpoints to your Google Drive, sets them to
"Anyone with the link can view", and automatically patches
download_checkpoints.py with the real file IDs.

Usage:
    pip install google-auth-oauthlib google-api-python-client tqdm
    python upload_to_gdrive.py

On first run a browser window opens for Google login (use neerajbhuvanmnb@gmail.com).
A token is saved to gdrive_token.json so future runs skip the login.

The script is safe to re-run — it skips files already uploaded to Drive.
"""

import io
import json
import os
import re
import sys
import zipfile

# ── dependency check ──────────────────────────────────────────────────────────
MISSING = []
for pkg in ("googleapiclient", "google_auth_oauthlib", "tqdm"):
    try:
        __import__(pkg)
    except ImportError:
        MISSING.append(pkg)

if MISSING:
    print("Missing dependencies. Install with:")
    print(f"  pip install google-auth-oauthlib google-api-python-client tqdm")
    sys.exit(1)

from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from tqdm import tqdm

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
CHECKPOINTS_DIR = os.path.join(ROOT, "agevision_backend", "checkpoints")
FADING_SRC = os.path.join(CHECKPOINTS_DIR, "fading",
                           "finetune_double_prompt_150_random")
FADING_ZIP = os.path.join(CHECKPOINTS_DIR, "fading", "fading_model.zip")
DOWNLOAD_SCRIPT = os.path.join(ROOT, "download_checkpoints.py")
TOKEN_FILE = os.path.join(ROOT, "gdrive_token.json")
DRIVE_FOLDER_NAME = "AgeVision-Checkpoints"

# Google Drive OAuth scopes
SCOPES = ["https://www.googleapis.com/auth/drive.file"]

# ── files to upload ───────────────────────────────────────────────────────────
# key → (local_path, gdrive_filename)
UPLOADS = {
    "sam_indian":      (os.path.join(CHECKPOINTS_DIR, "sam_indian_best.pt"),
                        "sam_indian_best.pt"),
    "sam_ffhq":        (os.path.join(CHECKPOINTS_DIR, "sam_ffhq_aging.pt"),
                        "sam_ffhq_aging.pt"),
    "hrfae":           (os.path.join(CHECKPOINTS_DIR, "hrfae_best.pth"),
                        "hrfae_best.pth"),
    "fast_aging":      (os.path.join(CHECKPOINTS_DIR, "fast_aging_gan.pth"),
                        "fast_aging_gan.pth"),
    "mivolo":          (os.path.join(CHECKPOINTS_DIR, "mivolo_indian",
                                     "mivolo_indian_best.pt"),
                        "mivolo_indian_best.pt"),
    "dlib_landmarks":  (os.path.join(CHECKPOINTS_DIR,
                                     "shape_predictor_68_face_landmarks.dat"),
                        "shape_predictor_68_face_landmarks.dat"),
    "fading":          (FADING_ZIP, "fading_model.zip"),
}


# ── helpers ───────────────────────────────────────────────────────────────────

def authenticate() -> object:
    """OAuth2 login. Opens browser on first run, reuses token after."""
    creds = None

    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            # Build a minimal OAuth client config inline (no client_secrets.json needed)
            # Uses Google's installed-app flow with the Drive file scope.
            print("\nOpening browser for Google login...")
            print("Sign in with: neerajbhuvanmnb@gmail.com\n")
            flow = InstalledAppFlow.from_client_config(
                {
                    "installed": {
                        "client_id": "YOUR_OAUTH_CLIENT_ID",
                        "client_secret": "YOUR_OAUTH_CLIENT_SECRET",
                        "redirect_uris": ["urn:ietf:wg:oauth:2.0:oob",
                                          "http://localhost"],
                        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                        "token_uri": "https://oauth2.googleapis.com/token",
                    }
                },
                SCOPES,
            )
            creds = flow.run_local_server(port=0)

        with open(TOKEN_FILE, "w") as f:
            f.write(creds.to_json())

    return build("drive", "v3", credentials=creds)


def get_or_create_folder(service, name: str) -> str:
    """Return the Drive folder ID for `name`, creating it if needed."""
    results = service.files().list(
        q=f"name='{name}' and mimeType='application/vnd.google-apps.folder' "
          f"and trashed=false",
        fields="files(id, name)",
    ).execute()

    files = results.get("files", [])
    if files:
        folder_id = files[0]["id"]
        print(f"  Using existing Drive folder: {name} ({folder_id})")
        return folder_id

    folder_meta = {
        "name": name,
        "mimeType": "application/vnd.google-apps.folder",
    }
    folder = service.files().create(body=folder_meta, fields="id").execute()
    folder_id = folder["id"]
    print(f"  Created Drive folder: {name} ({folder_id})")
    return folder_id


def file_exists_in_folder(service, folder_id: str, filename: str):
    """Return file ID if `filename` already exists in the Drive folder."""
    results = service.files().list(
        q=f"name='{filename}' and '{folder_id}' in parents and trashed=false",
        fields="files(id, name)",
    ).execute()
    files = results.get("files", [])
    return files[0]["id"] if files else None


def set_public(service, file_id: str):
    """Make a Drive file readable by anyone with the link."""
    service.permissions().create(
        fileId=file_id,
        body={"type": "anyone", "role": "reader"},
    ).execute()


def upload_file(service, local_path: str, filename: str,
                folder_id: str) -> str:
    """Resumable upload with tqdm progress. Returns file ID."""
    file_size = os.path.getsize(local_path)
    size_mb = file_size / (1024 * 1024)

    media = MediaFileUpload(
        local_path,
        mimetype="application/octet-stream",
        resumable=True,
        chunksize=8 * 1024 * 1024,  # 8 MB chunks
    )
    file_meta = {"name": filename, "parents": [folder_id]}
    request = service.files().create(body=file_meta, media_body=media,
                                     fields="id")

    print(f"\n  Uploading {filename} ({size_mb:.1f} MB)...")
    with tqdm(total=file_size, unit="B", unit_scale=True,
              unit_divisor=1024, desc=f"  {filename}") as bar:
        response = None
        while response is None:
            status, response = request.next_chunk()
            if status:
                bar.update(status.resumable_progress - bar.n)
        bar.update(file_size - bar.n)

    file_id = response["id"]
    print(f"  Done: {filename} → {file_id}")
    return file_id


def zip_fading_model():
    """Zip the fading model folder with progress. Skips if zip exists."""
    if os.path.exists(FADING_ZIP):
        print(f"  fading_model.zip already exists, skipping compression.")
        return

    if not os.path.isdir(FADING_SRC):
        print(f"  WARNING: FADING source folder not found: {FADING_SRC}")
        return

    print(f"\n  Zipping FADING model (this may take a few minutes)...")
    all_files = []
    for dirpath, _, filenames in os.walk(FADING_SRC):
        for fname in filenames:
            all_files.append(os.path.join(dirpath, fname))

    with zipfile.ZipFile(FADING_ZIP, "w", zipfile.ZIP_STORED,
                         allowZip64=True) as zf:
        with tqdm(total=len(all_files), unit="file",
                  desc="  Zipping FADING") as bar:
            for filepath in all_files:
                arcname = os.path.relpath(filepath, CHECKPOINTS_DIR)
                zf.write(filepath, arcname)
                bar.update(1)

    size_gb = os.path.getsize(FADING_ZIP) / (1024 ** 3)
    print(f"  Created fading_model.zip ({size_gb:.2f} GB)")


def patch_download_script(id_map: dict):
    """
    Replace placeholder IDs in download_checkpoints.py.
    Uses the comment anchors above each dict entry to match keys.
    """
    with open(DOWNLOAD_SCRIPT, "r", encoding="utf-8") as f:
        content = f.read()

    patched = content
    for key, file_id in id_map.items():
        # Pattern: the key name appears in the comment above the gdrive_id line
        # Replace "YOUR_GDRIVE_FILE_ID" only in the block for this key.
        # We match the block by looking for the key string followed shortly
        # by "YOUR_GDRIVE_FILE_ID" and replace just that occurrence.
        pattern = (
            r'("' + re.escape(key) + r'":\s*\{[^}]*?"gdrive_id":\s*)'
            r'"YOUR_GDRIVE_FILE_ID"'
        )
        replacement = r'\g<1>"' + file_id + '"'
        patched, count = re.subn(pattern, replacement, patched, flags=re.DOTALL)
        if count == 0:
            print(f"  WARNING: Could not patch key '{key}' in "
                  f"download_checkpoints.py — update manually.")

    with open(DOWNLOAD_SCRIPT, "w", encoding="utf-8") as f:
        f.write(patched)

    print(f"\n  Patched download_checkpoints.py with {len(id_map)} file IDs.")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("\nAgeVision — Google Drive Checkpoint Uploader")
    print("=" * 52)

    # 1. Zip FADING model folder
    print("\n[1/4] Preparing FADING model zip...")
    zip_fading_model()

    # 2. Authenticate
    print("\n[2/4] Authenticating with Google Drive...")
    print("      NOTE: You need to set up OAuth credentials first.")
    print("      See README section 'Setting up Google OAuth' below.\n")

    # Check for OAuth credentials
    client_secrets = os.path.join(ROOT, "gdrive_client_secrets.json")
    if not os.path.exists(client_secrets) and not os.path.exists(TOKEN_FILE):
        print("ERROR: No OAuth credentials found.")
        print("\nTo set up OAuth credentials:")
        print("  1. Go to https://console.cloud.google.com/")
        print("  2. Create a project → Enable Google Drive API")
        print("  3. Create OAuth 2.0 credentials (Desktop app type)")
        print("  4. Download JSON → save as 'gdrive_client_secrets.json' in project root")
        print("  5. Run this script again")
        sys.exit(1)

    # Load credentials from client_secrets.json if available
    if os.path.exists(client_secrets):
        creds = None
        if os.path.exists(TOKEN_FILE):
            creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                print("Opening browser for Google login...")
                print("Sign in with: neerajbhuvanmnb@gmail.com\n")
                flow = InstalledAppFlow.from_client_secrets_file(
                    client_secrets, SCOPES)
                creds = flow.run_local_server(port=0)
            with open(TOKEN_FILE, "w") as f:
                f.write(creds.to_json())

        service = build("drive", "v3", credentials=creds)
    else:
        # Token file exists — use it directly
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
        service = build("drive", "v3", credentials=creds)

    print("  Authenticated successfully.")

    # 3. Create/find Drive folder
    print(f"\n[3/4] Setting up Drive folder '{DRIVE_FOLDER_NAME}'...")
    folder_id = get_or_create_folder(service, DRIVE_FOLDER_NAME)

    # 4. Upload each file
    print("\n[4/4] Uploading checkpoints...")
    id_map = {}

    for key, (local_path, gdrive_name) in UPLOADS.items():
        if not os.path.exists(local_path):
            print(f"\n  [SKIP] {key}: file not found at {local_path}")
            continue

        existing_id = file_exists_in_folder(service, folder_id, gdrive_name)
        if existing_id:
            print(f"\n  [EXISTS] {gdrive_name} already in Drive ({existing_id})")
            id_map[key] = existing_id
            # Ensure it's public
            set_public(service, existing_id)
            continue

        file_id = upload_file(service, local_path, gdrive_name, folder_id)
        set_public(service, file_id)
        id_map[key] = file_id

    # 5. Patch download_checkpoints.py
    if id_map:
        patch_download_script(id_map)

    # 6. Print summary
    print("\n" + "=" * 52)
    print("UPLOAD COMPLETE — Share URLs:")
    print("=" * 52)
    for key, file_id in id_map.items():
        url = f"https://drive.google.com/file/d/{file_id}/view"
        print(f"  {key:16s}  {url}")

    folder_url = f"https://drive.google.com/drive/folders/{folder_id}"
    print(f"\n  Full folder: {folder_url}")
    print("\nNext steps:")
    print("  git add download_checkpoints.py")
    print('  git commit -m "feat: add real Google Drive IDs for checkpoints"')
    print("  git push")


if __name__ == "__main__":
    main()
