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
│  predictions) │         │  6.5 GB models  │
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
| Diffusion Aging | Stable Diffusion (prompt-to-prompt) |
| Face Detection | Caffe SSD, dlib, OpenCV Haar cascade |
| ML Frameworks | PyTorch, TensorFlow, ONNX Runtime |

---

## Project Structure

```
AgeVisionProject/
├── run.py                          # Standalone CLI pipeline entry point
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
│   │   ├── views/                  # 7 view modules (auth, predict, progress, history,
│   │   │                           #   analytics, settings, admin)
│   │   ├── age_predictor.py        # Prediction orchestration
│   │   ├── gan_progression.py      # GAN-based age progression
│   │   ├── mivolo_predictor.py     # MiVOLO age estimation
│   │   ├── insightface_predictor.py
│   │   ├── emotion_detector.py
│   │   ├── mongodb.py              # All MongoDB data access
│   │   ├── crypto.py               # Fernet encryption for sensitive data
│   │   ├── models.py               # Django ORM models
│   │   ├── serializers.py
│   │   ├── permissions.py
│   │   ├── urls.py                 # 27 API endpoints
│   │   ├── hrfae/                  # HRFAE GAN model
│   │   ├── mivolo/                 # MiVOLO model assets
│   │   ├── sam/                    # SAM model (50+ files)
│   │   ├── diffusion_aging/        # Stable diffusion aging
│   │   └── fast_aging/             # Fast GAN aging model
│   ├── age_progression/            # Alternative aging Django app
│   │   └── utils/                  # face_detector, age_estimator, age_progressor
│   └── checkpoints/                # ML model weights (gitignored, ~6.5 GB)
│       ├── sam_indian_best.pt      # SAM fine-tuned on Indian faces
│       ├── sam_ffhq_aging.pt       # SAM FFHQ general model
│       ├── hrfae_best.pth          # HRFAE GAN weights
│       ├── fast_aging_gan.pth      # Fast GAN weights
│       ├── mivolo_indian/          # MiVOLO age predictor
│       ├── fading/                 # Diffusion model weights
│       └── shape_predictor_68_face_landmarks.dat
│
├── agevision-frontend/             # Angular 19 web app
│   └── src/app/
│       ├── pages/                  # auth, dashboard, predict, progress,
│       │                           # history, analytics, admin, settings, batch-predict
│       ├── services/               # api.service.ts, auth.service.ts
│       └── models/                 # TypeScript interfaces
│
└── scripts/                        # Report generation utilities
    ├── build_final_report.py       # Generates Word (.docx) report
    ├── build_final_report_pdf.py   # Generates PDF report
    └── report_content.py           # Report content and figure definitions
```

---

## Prerequisites

- Python 3.9+ (3.13 recommended)
- Node.js 18+ and npm
- MongoDB 6.0+ (running locally or Atlas URI)
- CUDA-capable GPU (optional but recommended for inference speed)

---

## Setup

### 1. Clone & install Python dependencies

```bash
git clone https://github.com/NeerajBhuvan/AgeVisionProject.git
cd AgeVisionProject
pip install -r requirements.txt
```

### 2. Download model checkpoints

The `agevision_backend/checkpoints/` directory (~6.5 GB) is gitignored and must be set up manually. Place the following files:

```
agevision_backend/checkpoints/
├── sam_indian_best.pt              # SAM fine-tuned (Indian face dataset)
├── sam_ffhq_aging.pt               # SAM FFHQ pretrained
├── hrfae_best.pth                  # HRFAE GAN
├── fast_aging_gan.pth              # Fast GAN
├── shape_predictor_68_face_landmarks.dat   # dlib 68-point landmarks
├── mivolo_indian/                  # MiVOLO checkpoint folder
└── fading/                         # Diffusion model folder
    └── finetune_double_prompt_150_random/
```

### 3. Configure MongoDB

Start MongoDB locally or set your Atlas URI. By default the app connects to:
```
mongodb://localhost:27017/agevision
```

To use a custom URI, set it in `agevision_backend/agevision_backend/settings.py` under `MONGO_URI`.

### 4. Run Django migrations and start the backend

```bash
cd agevision_backend
python manage.py migrate
python manage.py runserver
```

Backend will be available at `http://localhost:8000`.

### 5. Start the Angular frontend

```bash
cd agevision-frontend
npm install
ng serve
```

Frontend will be available at `http://localhost:4200`.

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

# CPU only (no GPU)
python run.py --input ./images --output ./results --device cpu
```

**Output structure:**
```
results/
├── <name>_age20.jpg        # Aged images per target age
├── <name>_age40.jpg
├── <name>_grid.jpg         # Side-by-side comparison grid
├── metrics.json            # SSIM, PSNR, identity scores
└── report.html             # Visual HTML report
```

---

## Files Not in Git (Important — Must Be Set Up Manually)

These are gitignored because they are too large or environment-specific:

| Path | Size | Why Needed |
|---|---|---|
| `agevision_backend/checkpoints/` | ~6.5 GB | All ML models — app won't run without them |
| `agevision_backend/media/` | varies | User-uploaded images (generated at runtime) |
| `agevision_backend/datasets/` | 8.1 MB | MiVOLO Indian face training data |
| `age_vision_logo.png` | 6.9 MB | Logo used in generated reports |
| `screenshots/` | 42 MB | Figure screenshots for Word/PDF report generation |
| `images/` | 2.3 MB | Test face samples for CLI pipeline (`python run.py --input ./images`) |
| `agevision_backend/db.sqlite3` | varies | SQLite DB (used only as Django fallback) |

---

## Running Tests

```bash
cd agevision_backend

# Auth system
python test_auth.py

# Age prediction accuracy
python test_age_prediction.py

# GAN accuracy evaluation
python test_gan_accuracy.py

# End-to-end API
python test_api_e2e.py

# MongoDB integration
python test_mongodb_e2e.py
```

---

## Generating the Project Report

```bash
# Word (.docx) report
python scripts/build_final_report.py

# PDF report
python scripts/build_final_report_pdf.py
```

Requires `screenshots/` directory and `age_vision_logo.png` to be present for figures. Falls back to placeholder boxes if images are missing.
