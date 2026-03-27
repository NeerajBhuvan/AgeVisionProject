"""
Fast-AgingGAN Inference Wrapper
================================
Lightweight face aging using CycleGAN generator trained on UTKFace
(includes Indian/South Asian faces).

Young → Old transformation in a single forward pass (~11MB model).
Does NOT support specific target age — always ages to "elderly".
"""

import logging
import os

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from .models import Generator

logger = logging.getLogger("agevision.fast_aging")

IMG_SIZE = 512

FAST_AGING_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
])

# Default checkpoint path
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))  # agevision_backend/
DEFAULT_CHECKPOINT = os.path.join(_BASE_DIR, 'checkpoints', 'fast_aging_gan.pth')


class FastAgingInference:
    """
    Fast-AgingGAN inference wrapper.

    Interface:
      - load() -> bool
      - is_ready -> bool
      - transform_face(face_crop_bgr, target_age) -> np.ndarray (BGR)

    Note: target_age is accepted for API compatibility but ignored —
    the model always produces an "elderly" transformation.
    """

    def __init__(self, checkpoint_path: str = None, device: str = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.checkpoint_path = checkpoint_path or DEFAULT_CHECKPOINT
        self._loaded = False
        self.net = None

    def load(self) -> bool:
        """Load Fast-AgingGAN generator weights. Returns True on success."""
        if self._loaded:
            return True

        if not os.path.isfile(self.checkpoint_path):
            logger.error("Fast-AgingGAN checkpoint not found: %s", self.checkpoint_path)
            return False

        try:
            logger.info("Loading Fast-AgingGAN from %s ...", self.checkpoint_path)

            self.net = Generator(ngf=32, n_residual_blocks=9)
            state_dict = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
            self.net.load_state_dict(state_dict)
            self.net.eval()

            if self.device == "cuda":
                self.net.cuda()

            self._loaded = True
            logger.info("Fast-AgingGAN loaded on %s", self.device)
            return True

        except Exception as e:
            logger.error("Failed to load Fast-AgingGAN: %s", e)
            self._loaded = False
            return False

    @property
    def is_ready(self) -> bool:
        return self._loaded

    @torch.no_grad()
    def transform_face(self, face_crop_bgr: np.ndarray, target_age: int = 70) -> np.ndarray:
        """
        Age a face crop to elderly appearance.

        Args:
            face_crop_bgr: BGR face image (any size, will be resized to 512x512).
            target_age: Ignored — model always produces elderly output.

        Returns:
            Aged face as BGR numpy array (512x512).
        """
        if not self._loaded:
            raise RuntimeError("Fast-AgingGAN not loaded. Call load() first.")

        # BGR -> RGB -> PIL
        face_rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
        face_pil = Image.fromarray(face_rgb)

        # Preprocess: resize to 512x512, normalize to [-1, 1]
        input_tensor = FAST_AGING_TRANSFORM(face_pil).unsqueeze(0).to(self.device)

        # Forward pass
        output_tensor = self.net(input_tensor)

        # Post-process: [-1, 1] -> [0, 255] BGR
        output_np = output_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
        output_np = ((output_np + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
        output_bgr = cv2.cvtColor(output_np, cv2.COLOR_RGB2BGR)

        return output_bgr
