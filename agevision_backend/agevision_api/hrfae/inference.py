"""
HRFAE Inference Pipeline
========================
Loads a trained HRFAE checkpoint and applies age transformation
to a single face image.

Key changes from generic inference:
  • Model expects [0, 1] input (not [-1, 1])
  • Age is passed as integer (0–100), not normalized float
  • Model uses skip connections internally

Usage (standalone):
    python -m agevision_api.hrfae.inference --image face.jpg --age 65
"""

import os
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from .model import HRFAE


# ── Default paths ──
_DEFAULT_CKPT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..', 'checkpoints'
)
_DEFAULT_CKPT = os.path.join(_DEFAULT_CKPT_DIR, 'hrfae_best.pth')

IMG_SIZE = 256

# Official HRFAE normalization (mean subtraction, no std division)
HRFAE_MEAN = [0.48501961, 0.45795686, 0.40760392]

# Pre-processing: ToTensor [0,1] → subtract mean
_to_tensor = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),          # scales to [0, 1]
    transforms.Normalize(mean=HRFAE_MEAN, std=[1, 1, 1]),
])

_face_cascade = None


def _get_face_cascade():
    global _face_cascade
    if _face_cascade is None:
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        _face_cascade = cv2.CascadeClassifier(cascade_path)
    return _face_cascade


def _detect_and_crop_face(img_bgr: np.ndarray):
    """Detect the largest face and crop with padding."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    cascade = _get_face_cascade()
    faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))

    if len(faces) == 0:
        # Fall back to center crop
        h, w = img_bgr.shape[:2]
        s = min(h, w)
        y0, x0 = (h - s) // 2, (w - s) // 2
        return img_bgr[y0:y0 + s, x0:x0 + s], (x0, y0, s, s)

    # Largest face
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    x, y, w, h = faces[0]

    # Expand ROI by 40%/50% for forehead/chin
    pad_w = int(w * 0.4)
    pad_h = int(h * 0.5)
    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(img_bgr.shape[1], x + w + pad_w)
    y2 = min(img_bgr.shape[0], y + h + pad_h)

    # Make square
    crop_w, crop_h = x2 - x1, y2 - y1
    side = max(crop_w, crop_h)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    x1 = max(0, cx - side // 2)
    y1 = max(0, cy - side // 2)
    x2 = min(img_bgr.shape[1], x1 + side)
    y2 = min(img_bgr.shape[0], y1 + side)

    return img_bgr[y1:y2, x1:x2], (x1, y1, x2 - x1, y2 - y1)


class HRFAEInference:
    """
    High-level inference wrapper for the HRFAE model.

    Handles face detection, cropping, pre-processing, model inference,
    and post-processing back to a full-resolution image.
    """

    def __init__(self, checkpoint_path: str = None, device: str = None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model = HRFAE()
        self.model.eval()
        self.model.to(self.device)

        self.checkpoint_path = checkpoint_path or _DEFAULT_CKPT
        self._loaded = False

    @property
    def is_ready(self) -> bool:
        """Check if the model weights are loaded and ready."""
        if not self._loaded:
            self._try_load()
        return self._loaded

    def _try_load(self):
        """Attempt to load the checkpoint."""
        if os.path.isfile(self.checkpoint_path):
            try:
                self.model.load_checkpoint(self.checkpoint_path,
                                           device=str(self.device))
                self.model.eval()
                self._loaded = True
            except Exception as e:
                print(f"[HRFAE] Failed to load checkpoint: {e}")
                self._loaded = False
        else:
            self._loaded = False

    @torch.no_grad()
    def transform(self, image_path: str, target_age: int) -> dict:
        """
        Apply age transformation to a face image.

        Args:
            image_path : Path to the input face image.
            target_age : Target age (0–100).

        Returns:
            dict with:
                - output_image : np.ndarray (BGR, full-resolution)
                - face_crop    : np.ndarray (BGR, 256×256 crop of input face)
                - aged_crop    : np.ndarray (BGR, 256×256 aged crop)
                - face_rect    : (x, y, w, h) of the detected face
        """
        if not self.is_ready:
            raise RuntimeError(
                "HRFAE model not loaded. Train the model first using:\n"
                "  python -m agevision_api.hrfae.train"
            )

        # Read image
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise ValueError(f"Cannot read image: {image_path}")

        orig_h, orig_w = img_bgr.shape[:2]

        # Detect and crop face
        face_crop, face_rect = _detect_and_crop_face(img_bgr)

        # Convert to PIL → tensor  (produces [0, 1] range)
        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        face_pil = Image.fromarray(face_rgb)
        input_tensor = _to_tensor(face_pil).unsqueeze(0).to(self.device)

        # Target age as integer tensor [B]
        age_tensor = torch.tensor([int(min(max(target_age, 0), 100))],
                                  dtype=torch.long, device=self.device)

        # Forward pass
        output_tensor = self.model(input_tensor, age_tensor)

        # Apply histogram transfer (match output colors to input)
        output_tensor = self._hist_transfer(output_tensor[0], input_tensor[0])

        # Post-process: add mean back, clamp, tensor → numpy
        aged_crop = self._tensor_to_cv2(output_tensor)

        # Paste the aged crop back into the original image
        output_image = self._paste_back(
            img_bgr.copy(), aged_crop, face_rect, orig_w, orig_h
        )

        return {
            'output_image': output_image,
            'face_crop': cv2.resize(face_crop, (IMG_SIZE, IMG_SIZE)),
            'aged_crop': aged_crop,
            'face_rect': face_rect,
        }

    @staticmethod
    def _hist_transfer(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Histogram matching — maps output color distribution to input.
        This is the official HRFAE hist_transform function."""
        c, h, w = source.size()
        s_t = source.clone().view(c, -1)
        t_t = target.view(c, -1)
        s_t_sorted, s_t_indices = torch.sort(s_t)
        t_t_sorted, t_t_indices = torch.sort(t_t)
        for i in range(c):
            s_t[i, s_t_indices[i]] = t_t_sorted[i]
        return s_t.view(c, h, w)

    @staticmethod
    def _tensor_to_cv2(tensor: torch.Tensor) -> np.ndarray:
        """Convert a mean-subtracted tensor to a BGR uint8 numpy array.
        Adds the HRFAE mean back and clamps to [0, 1]."""
        img = tensor.cpu().clone()
        # Add mean back (inverse of preprocessing normalization)
        img[0] += HRFAE_MEAN[0]
        img[1] += HRFAE_MEAN[1]
        img[2] += HRFAE_MEAN[2]
        img = img.clamp(0, 1).numpy()
        img = (img * 255.0).astype(np.uint8)  # [0, 255]
        img = img.transpose(1, 2, 0)           # C, H, W → H, W, C
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    @staticmethod
    def _paste_back(original: np.ndarray, aged_crop: np.ndarray,
                    face_rect: tuple, orig_w: int, orig_h: int) -> np.ndarray:
        """Resize the aged crop and blend it back into the original."""
        x, y, w, h = face_rect
        aged_resized = cv2.resize(aged_crop, (w, h))

        # Create a soft elliptical mask for blending
        mask = np.zeros((h, w), dtype=np.float32)
        cv2.ellipse(mask,
                    (w // 2, h // 2),
                    (int(w * 0.42), int(h * 0.42)),
                    0, 0, 360, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (31, 31), 10)
        mask3 = np.stack([mask] * 3, axis=-1)

        # Blend
        roi = original[y:y + h, x:x + w].astype(np.float32)
        blended = aged_resized.astype(np.float32) * mask3 + roi * (1 - mask3)
        original[y:y + h, x:x + w] = np.clip(blended, 0, 255).astype(np.uint8)

        return original


# ─── CLI entry point ───
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='HRFAE Age Transformation')
    parser.add_argument('--image', required=True, help='Input face image path')
    parser.add_argument('--age', type=int, required=True, help='Target age (0-100)')
    parser.add_argument('--checkpoint', default=None, help='Model checkpoint path')
    parser.add_argument('--output', default='output_aged.jpg', help='Output path')
    args = parser.parse_args()

    model = HRFAEInference(checkpoint_path=args.checkpoint)
    result = model.transform(args.image, args.age)
    cv2.imwrite(args.output, result['output_image'])
    print(f"Saved aged image to: {args.output}")
