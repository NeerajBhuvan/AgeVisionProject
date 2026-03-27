"""
HRFAE Age Progression Model Wrapper
=====================================
Standalone wrapper for the HRFAE GAN model.
Handles model loading, inference, and age transformation
without Django dependencies.
"""

import logging
import sys
import os
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from pathlib import Path

logger = logging.getLogger("age_pipeline.model")

# Add the backend to sys.path so we can import the HRFAE model
_BACKEND_DIR = str(Path(__file__).resolve().parent.parent / "agevision_backend")
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

# Import the HRFAE model architecture
from agevision_api.hrfae.model import HRFAE

# Default checkpoint path
DEFAULT_CHECKPOINT = Path(__file__).resolve().parent.parent / "agevision_backend" / "checkpoints" / "hrfae_best.pth"

# HRFAE preprocessing constants
IMG_SIZE = 256
HRFAE_MEAN = [0.48501961, 0.45795686, 0.40760392]

_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=HRFAE_MEAN, std=[1, 1, 1]),
])


class AgeProgressionModel:
    """Standalone HRFAE model wrapper for age progression inference."""

    def __init__(self, checkpoint_path: str = None, device: str = None):
        """Initialize the model.

        Args:
            checkpoint_path: Path to HRFAE checkpoint. Uses default if None.
            device: 'cuda' or 'cpu'. Auto-detects if None.
        """
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                logger.info("Using GPU: %s", torch.cuda.get_device_name(0))
            else:
                self.device = torch.device("cpu")
                logger.info("Using CPU (CUDA not available)")
        else:
            self.device = torch.device(device)

        self.checkpoint_path = str(checkpoint_path or DEFAULT_CHECKPOINT)
        self.model = None
        self._loaded = False

    def load(self) -> bool:
        """Load model weights. Returns True on success."""
        if self._loaded:
            return True

        if not os.path.isfile(self.checkpoint_path):
            logger.error("Checkpoint not found: %s", self.checkpoint_path)
            return False

        try:
            self.model = HRFAE()
            self.model.load_checkpoint(self.checkpoint_path, device=str(self.device))
            self.model.to(self.device)
            self.model.eval()
            self._loaded = True
            logger.info("HRFAE model loaded from %s", self.checkpoint_path)
            return True
        except Exception as e:
            logger.error("Failed to load HRFAE model: %s", e)
            self._loaded = False
            return False

    @property
    def is_ready(self) -> bool:
        return self._loaded

    @torch.no_grad()
    def transform_face(self, face_crop_bgr: np.ndarray, target_age: int) -> np.ndarray:
        """Transform a pre-cropped face image to target age.

        Args:
            face_crop_bgr: BGR face crop (any size, will be resized to 256x256).
            target_age: Target age (0-100).

        Returns:
            Aged face crop as BGR numpy array (256x256).
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Clamp age to valid range
        target_age = max(0, min(100, target_age))

        # Convert BGR -> RGB -> PIL -> tensor
        face_rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
        face_pil = Image.fromarray(face_rgb)
        input_tensor = _transform(face_pil).unsqueeze(0).to(self.device)

        # Age tensor
        age_tensor = torch.tensor([target_age], dtype=torch.long, device=self.device)

        # Forward pass
        output_tensor = self.model(input_tensor, age_tensor)

        # Histogram transfer for color consistency
        output_tensor = self._hist_transfer(output_tensor[0], input_tensor[0])

        # Convert back to BGR numpy
        return self._tensor_to_bgr(output_tensor)

    @torch.no_grad()
    def transform_batch(self, face_crops: list, target_ages: list) -> list:
        """Transform a batch of face crops to target ages.

        Args:
            face_crops: List of BGR face crops.
            target_ages: List of target ages.

        Returns:
            List of aged face crops (BGR numpy arrays).
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        results = []
        # Process one at a time to manage memory on CPU
        for crop, age in zip(face_crops, target_ages):
            try:
                aged = self.transform_face(crop, age)
                results.append(aged)
            except Exception as e:
                logger.error("Failed to transform face to age %d: %s", age, e)
                results.append(crop)  # Fallback to original
        return results

    @staticmethod
    def _hist_transfer(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Histogram matching to preserve color distribution."""
        c, h, w = source.size()
        s_t = source.clone().view(c, -1)
        t_t = target.view(c, -1)
        s_t_sorted, s_t_indices = torch.sort(s_t)
        t_t_sorted, _ = torch.sort(t_t)
        for i in range(c):
            s_t[i, s_t_indices[i]] = t_t_sorted[i]
        return s_t.view(c, h, w)

    @staticmethod
    def _tensor_to_bgr(tensor: torch.Tensor) -> np.ndarray:
        """Convert mean-subtracted tensor to BGR uint8 numpy array."""
        img = tensor.cpu().clone()
        img[0] += HRFAE_MEAN[0]
        img[1] += HRFAE_MEAN[1]
        img[2] += HRFAE_MEAN[2]
        img = img.clamp(0, 1).numpy()
        img = (img * 255.0).astype(np.uint8)
        img = img.transpose(1, 2, 0)  # CHW -> HWC
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
