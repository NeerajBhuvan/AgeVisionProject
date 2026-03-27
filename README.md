# AgeVision — Age Progression Pipeline

End-to-end age progression system using HRFAE (High Resolution Face Age Editing) GAN.

## Quick Start

```bash
# Process images with default age targets (20, 40, 60, 80)
python run.py --input ./images --output ./results --ages 20,40,60,80

# Single image
python run.py --input photo.jpg --output ./results --ages 30,50,70

# Skip deep identity metrics (faster)
python run.py --input ./images --output ./results --ages 20,40,60,80 --no-deep

# Verbose logging
python run.py --input ./images --output ./results --ages 20,40,60,80 --verbose
```

## CLI Options

| Flag | Description |
|------|-------------|
| `--input, -i` | Input image file or directory (required) |
| `--output, -o` | Output directory (default: `./results`) |
| `--ages, -a` | Comma-separated target ages (default: `20,40,60,80`) |
| `--checkpoint, -c` | HRFAE checkpoint path (auto-detected) |
| `--device, -d` | `cuda` or `cpu` (auto-detected) |
| `--current-age` | Subject's current age (for effect scaling) |
| `--no-deep` | Skip FaceNet identity metrics |
| `--no-report` | Skip HTML report generation |
| `--verbose, -v` | Debug logging |

## Pipeline Stages

1. **Face Detection** — Caffe SSD + Haar cascade fallback
2. **Preprocessing** — Crop, align, resize to 256×256
3. **HRFAE GAN Inference** — Age-conditioned face transformation
4. **Post-Processing** — Blend back, color correct, enhance aging effects
5. **Evaluation** — SSIM, PSNR, identity score, age accuracy
6. **Reporting** — HTML visual report + JSON metrics

## Outputs

```
results/
├── <name>_age<N>.jpg    # Individual aged images
├── <name>_grid.jpg      # Side-by-side comparison grid
├── metrics.json         # Full evaluation metrics
├── report.html          # Visual HTML report
└── pipeline.log         # Processing log
```

## Project Structure

```
age_pipeline/
├── __init__.py          # Package init
├── detector.py          # Face detection & validation
├── model.py             # HRFAE model wrapper
├── postprocess.py       # Blending, color correction, enhancement
├── evaluator.py         # SSIM, PSNR, identity, age estimation
└── report.py            # HTML report generator

run.py                   # CLI entry point
download_samples.py      # Generate/download test images
```

## Model

Uses HRFAE (InterDigitalInc/HRFAE) with pretrained checkpoint at:
`agevision_backend/checkpoints/hrfae_best.pth`

Auto-detects GPU/CPU. Falls back to CPU if CUDA unavailable.
