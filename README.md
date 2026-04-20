# AgeVision — Age-Invariant Face Recognition & Retrieval

A full-stack AI system for age prediction, age progression, and face recognition that works across different age groups. Built as an MCA project for Anna University Centre for Distance Education.

---

## Features

- **Age Prediction** — Predict age, gender, emotion, and race from a face image using MiVOLO + InsightFace ensemble
- **Age Progression** — Generate realistic aged/de-aged versions of a face (GAN + optional diffusion model)
- **Batch Processing** — Process multiple images in one request
- **Camera Stream** — Real-time age prediction from webcam feed
- **Admin Panel** — User management, system health monitoring, analytics dashboard
- **Prediction History** — Per-user history of all predictions and progressions
- **Secure Auth** — JWT authentication with encrypted storage (Fernet), password recovery
- **CLI Pipeline** — Standalone command-line tool for bulk age progression with HTML report output

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Angular 19 Frontend  (agevision-frontend/)             │
│  Auth · Dashboard · Predict · Progress · Admin          │
└───────────────────┬─────────────────────────────────────┘
                    │ REST API (JWT)
┌───────────────────▼─────────────────────────────────────┐
│  Django 5.2 REST API  (agevision_backend/)              │
│                                                         │
│  agevision_api/          age_progression/               │
│  ├─ age_predictor.py     └─ (alternative aging app)     │
│  ├─ gan_progression.py                                  │
│  ├─ mivolo_predictor.py                                 │
│  ├─ insightface_predictor.py                            │
│  └─ mongodb.py (data layer)                             │
└───────┬───────────────────────────┬─────────────────────┘
        │                           │
┌───────▼───────┐         ┌─────────▼──────┐
│   MongoDB     │         │  ML Checkpoints │
│  (user data,  │         │  checkpoints/   │
│  predictions) │         │  ~6.5 GB        │
└───────────────┘         └────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Angular 19, Bootstrap 5, Chart.js, jsPDF |
| Backend | Django 5.2, Django REST Framework, SimpleJWT |
| Database | MongoDB (primary), SQLite (fallback) |
| Age Prediction | MiVOLO, InsightFace, YOLO |
| Age Progression | HRFAE GAN, SAM (Style-based Aging Model) |
| Diffusion Aging | Stable Diffusion (FADING, prompt-to-prompt) |
| Face Detection | Caffe SSD, dlib, OpenCV Haar cascade |
| ML Frameworks | PyTorch, TensorFlow, ONNX Runtime |

---

## Project Structure

```
AgeVisionProject/
├── run.py                          # Standalone CLI pipeline entry point
├── download_checkpoints.py         # Downloads all models from Google Drive
├── requirements.txt                # Python dependencies (108 packages)
├── requirements_pipeline.txt       # Pipeline-only deps
│
├── age_pipeline/                   # Standalone CLI age progression package
│   ├── detector.py                 # Face detection (Caffe SSD + Haar fallback)
│   ├── model.py                    # HRFAE model wrapper
│   ├── postprocess.py              # Blending, color correction
│   ├── evaluator.py                # SSIM, PSNR, identity metrics
│   └── report.py                   # HTML report generator
│
├── agevision_backend/              # Django REST API
│   ├── manage.py
│   ├── agevision_backend/          # Django project config
│   │   ├── settings.py
│   │   └── urls.py
│   ├── agevision_api/              # Main API app
│   │   ├── views/                  # 7 view modules (auth, predict, progress,
│   │   │                           #   history, analytics, settings, admin)
│   │   ├── age_predictor.py        # Prediction orchestration
│   │   ├── gan_progression.py      # GAN-based age progression
│   │   ├── mivolo_predictor.py     # MiVOLO age estimation
│   │   ├── insightface_predictor.py
│   │   ├── emotion_detector.py
│   │   ├── mongodb.py              # All MongoDB data access
│   │   ├── crypto.py               # Fernet encryption
│   │   ├── hrfae/                  # HRFAE GAN model
│   │   ├── mivolo/                 # MiVOLO model assets
│   │   ├── sam/                    # SAM model (50+ files)
│   │   ├── diffusion_aging/        # FADING Stable Diffusion aging
│   │   └── fast_aging/             # Fast GAN aging model
│   ├── age_progression/            # Alternative aging Django app
│   └── checkpoints/                # ML model weights (gitignored, ~6.5 GB)
│
├── agevision-frontend/             # Angular 19 web app
│   └── src/app/
│       ├── pages/                  # auth, dashboard, predict, progress,
│       │                           # history, analytics, admin, settings, batch-predict
│       ├── services/               # api.service.ts, auth.service.ts
│       └── models/                 # TypeScript interfaces
│
└── scripts/                        # Report generation utilities
    ├── build_final_report.py
    ├── build_final_report_pdf.py
    └── report_content.py
```

---

## Prerequisites

- Python 3.9+ (3.13 recommended)
- Node.js 18+ and npm
- MongoDB 6.0+
- CUDA GPU recommended (GTX 1080 Ti / RTX 3080 or better for diffusion model)

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/NeerajBhuvan/AgeVisionProject.git
cd AgeVisionProject
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Download model checkpoints

```bash
python download_checkpoints.py
```

This downloads all models from Google Drive into `agevision_backend/checkpoints/`.  
See the [Model Checkpoints](#model-checkpoints) section below for details on each model.

### 4. Configure MongoDB

Start MongoDB locally or use Atlas. Default connection:
```
mongodb://localhost:27017/agevision
```
To use a custom URI, edit `MONGO_URI` in `agevision_backend/agevision_backend/settings.py`.

### 5. Start the backend

```bash
cd agevision_backend
python manage.py migrate
python manage.py runserver
```

Backend runs at `http://localhost:8000`.

### 6. Start the frontend

```bash
cd agevision-frontend
npm install
ng serve
```

Frontend runs at `http://localhost:4200`.

---

## Model Checkpoints

All model weights are stored in `agevision_backend/checkpoints/` (gitignored).  
They are hosted on Google Drive and downloaded automatically by `download_checkpoints.py`.

### Checkpoint Files

| Model | File | Size | Required | Google Drive |
|---|---|---|---|---|
| SAM (Indian) | `sam_indian_best.pt` | 2.2 GB | Yes | [Download](https://drive.google.com/file/d/1kQbJlKFv3MxnmYbWzp0BFM3jJcPshQMy/view) |
| SAM (FFHQ) | `sam_ffhq_aging.pt` | 2.2 GB | Yes | [Download](https://drive.google.com/file/d/1u_8VUyZlRUPJik5zUC2ZbrjmPNaEWvKI/view) |
| HRFAE GAN | `hrfae_best.pth` | 50 MB | Yes | [Download](https://drive.google.com/file/d/1ro-icYhFKvTuvl7pfEVSY-Hc6dp8kp31/view) |
| Fast-AgingGAN | `fast_aging_gan.pth` | 11 MB | No | [Download](https://drive.google.com/file/d/1e2yxCtx07EK7WdOD1dy9-h68GenS_i5Y/view) |
| MiVOLO | `mivolo_indian/mivolo_indian_best.pt` | 110 MB | Yes | [Download](https://drive.google.com/file/d/1z6wmcV21aPIAOY6WqNLS9rOzQW9EtzYA/view) |
| dlib Landmarks | `shape_predictor_68_face_landmarks.dat` | 96 MB | Yes | [Download](https://drive.google.com/file/d/1ro-AaviWt3y7irUGx-lSRGzfDV1uHTnD/view) |
| FADING (diffusion) | `fading/` (folder, zipped) | 5.2 GB | No | [Download](https://drive.google.com/file/d/1tWV_OQYtUSd2HtnDbHj-M1zqOR9p--Xc/view) |

### FADING Diffusion Model — Special Notes

The FADING model (`fading/`) is a fine-tuned Stable Diffusion pipeline and has three execution modes:

| Scenario | Behaviour |
|---|---|
| CUDA GPU with 6–8 GB VRAM | Loads locally from `checkpoints/fading/` |
| No local GPU, Modal configured | Sends images to Modal serverless cloud GPU |
| No GPU, no Modal | Skipped; falls back to SAM → GAN → OpenCV |

**To use Modal cloud GPU (no local GPU required):**

1. Create a free account at [modal.com](https://modal.com)
2. Deploy the FADING endpoint from your Modal workspace
3. Set the endpoint URL as an environment variable:
   ```bash
   export FADING_MODAL_ENDPOINT=https://your-workspace--fading-inference.modal.run
   ```
   Or add it to Django settings:
   ```python
   # agevision_backend/agevision_backend/settings.py
   FADING_MODAL_ENDPOINT = "https://your-workspace--fading-inference.modal.run"
   ```

### Age Progression Fallback Chain

The system automatically falls back if a model is unavailable:

```
SAM Indian  →  SAM FFHQ  →  Fast-AgingGAN  →  FADING  →  Rule-based OpenCV
 (best)                                        (cloud)       (no GPU needed)
```

This means the app works on any machine — quality degrades gracefully based on available hardware.

### How to Upload Checkpoints to Google Drive

1. Open [Google Drive](https://drive.google.com) and create a folder called `AgeVision-Checkpoints`
2. Upload each file listed in the table above
3. For the FADING model — zip the entire `fading/finetune_double_prompt_150_random/` folder first, name it `fading_model.zip`, then upload
4. For each uploaded file: **right-click → Share → change to "Anyone with the link" → Viewer**
5. Copy the share link. The file ID is the string between `/d/` and `/view`:
   ```
   https://drive.google.com/file/d/1abc123XYZ.../view
                                    ^^^^^^^^^^^^
                                    This is the ID
   ```
6. Replace `YOUR_GDRIVE_FILE_ID` in `download_checkpoints.py` and in the table above
7. Commit the updated `download_checkpoints.py`

---

## API Endpoints

| Category | Endpoint | Method | Description |
|---|---|---|---|
| Auth | `/api/auth/register/` | POST | Register new user |
| Auth | `/api/auth/login/` | POST | Login, returns JWT |
| Auth | `/api/auth/profile/` | GET/PUT | View/update profile |
| Auth | `/api/auth/forgot-password/` | POST | Initiate password reset |
| Predict | `/api/predict/` | POST | Predict age from image |
| Predict | `/api/predict/camera/` | POST | Predict from camera frame |
| Predict | `/api/predict/batch/` | POST | Batch predict multiple images |
| Progress | `/api/progress/` | POST | Generate aged face |
| Progress | `/api/progress/stream/` | GET | SSE stream for live progress |
| History | `/api/history/` | GET | User prediction history |
| History | `/api/history/<id>/` | DELETE | Delete a history entry |
| Analytics | `/api/analytics/` | GET | Usage analytics |
| Settings | `/api/settings/` | GET/PUT | User preferences |
| Admin | `/api/admin/dashboard/` | GET | Admin stats |
| Admin | `/api/admin/users/` | GET | All users |
| Admin | `/api/admin/users/<id>/suspend/` | POST | Suspend user |
| Admin | `/api/admin/system-health/` | GET | System health check |

---

## CLI Pipeline (Standalone)

Use `run.py` for bulk age progression without the web app:

```bash
# Process a directory of images at ages 20, 40, 60, 80
python run.py --input ./images --output ./results --ages 20,40,60,80

# Single image, specify current age for better scaling
python run.py --input photo.jpg --output ./results --ages 30,50,70 --current-age 25

# Faster run without deep identity metrics
python run.py --input ./images --output ./results --no-deep

# CPU only
python run.py --input ./images --output ./results --device cpu
```

**Output structure:**
```
results/
├── <name>_age20.jpg        # Aged image per target age
├── <name>_age40.jpg
├── <name>_grid.jpg         # Side-by-side comparison grid
├── metrics.json            # SSIM, PSNR, identity scores
└── report.html             # Visual HTML report
```

---

## Running Tests

```bash
cd agevision_backend
python test_auth.py             # Auth system
python test_age_prediction.py   # Age prediction accuracy
python test_gan_accuracy.py     # GAN progression accuracy
python test_api_e2e.py          # End-to-end API
python test_mongodb_e2e.py      # MongoDB integration
```

---

## Files Not in Git

These are gitignored and must be set up manually or downloaded:

| Path | Size | How to Get |
|---|---|---|
| `agevision_backend/checkpoints/` | ~6.5 GB | Run `python download_checkpoints.py` |
| `agevision_backend/media/` | varies | Created automatically at runtime |
| `agevision_backend/datasets/` | 8.1 MB | Training data — not needed to run the app |
| `age_vision_logo.png` | 6.9 MB | Needed only for report generation |
| `screenshots/` | 42 MB | Needed only for report generation |
| `images/` | 2.3 MB | Test samples for `python run.py --input ./images` |
| `node_modules/` | 409 MB | Restored automatically by `npm install` |

---

## Generating the Project Report

```bash
python scripts/build_final_report.py      # Word (.docx)
python scripts/build_final_report_pdf.py  # PDF
```

Requires `screenshots/` and `age_vision_logo.png` to be present for figures. Falls back to placeholder boxes if images are missing.
