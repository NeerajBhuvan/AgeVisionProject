"""
SAM (Style-based Age Manipulation) Inference Wrapper
=====================================================
Loads a SAM checkpoint (pSp encoder + StyleGAN2 decoder) and applies
age transformation. Supports both FFHQ-pretrained and Indian-finetuned
checkpoints.

Interface matches AgeProgressionModel for drop-in use in GANProgressionPipeline.
"""

import logging
import os
from argparse import Namespace

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

logger = logging.getLogger("agevision.sam")

# SAM preprocessing constants
IMG_SIZE = 256
SAM_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


class SAMInference:
    """
    High-level SAM model wrapper for age progression inference.

    Matches the interface of AgeProgressionModel:
      - load() -> bool
      - is_ready -> bool
      - transform_face(face_crop_bgr, target_age) -> np.ndarray (BGR)
    """

    # Separate singleton instances per variant
    _instances = {}  # {'indian': SAMInference, 'ffhq': SAMInference}

    def __init__(self, checkpoint_path: str = None, device: str = None,
                 variant: str = 'ffhq'):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.variant = variant

        if checkpoint_path is None:
            from .configs.paths_config import model_paths
            ffhq_path = model_paths.get('sam_ffhq_aging', '')
            indian_path = model_paths.get('sam_indian', '')

            if variant == 'indian':
                # Indian-first priority
                if os.path.isfile(indian_path):
                    checkpoint_path = indian_path
                elif os.path.isfile(ffhq_path):
                    checkpoint_path = ffhq_path
            else:
                # FFHQ-first priority
                if os.path.isfile(ffhq_path):
                    checkpoint_path = ffhq_path
                elif os.path.isfile(indian_path):
                    checkpoint_path = indian_path

        self.checkpoint_path = checkpoint_path
        self._loaded = False
        self.net = None
        self._opts = None

        # dlib predictor for FFHQ alignment (lazy-loaded)
        self._dlib_predictor = None

    @classmethod
    def get_instance(cls, variant: str = 'ffhq') -> 'SAMInference':
        """Get or create a singleton SAMInference for the given variant."""
        if variant not in cls._instances:
            cls._instances[variant] = cls(variant=variant)
        return cls._instances[variant]

    def load(self) -> bool:
        """Load SAM model weights. Returns True on success."""
        if self._loaded:
            return True

        if not self.checkpoint_path or not os.path.isfile(self.checkpoint_path):
            logger.error("SAM checkpoint not found: %s", self.checkpoint_path)
            return False

        try:
            logger.info("Loading SAM checkpoint from %s ...", self.checkpoint_path)

            # Load checkpoint and extract opts
            ckpt = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
            opts = ckpt["opts"]
            opts["checkpoint_path"] = self.checkpoint_path
            opts["device"] = self.device
            # Ensure required opts have defaults (guard against incomplete checkpoints)
            opts.setdefault("input_nc", 4)
            opts.setdefault("output_size", 1024)
            opts.setdefault("start_from_latent_avg", False)
            opts.setdefault("start_from_encoded_w_plus", True)
            opts.setdefault("stylegan_weights", "")
            opts.setdefault("pretrained_psp_path", "")
            self._opts = Namespace(**opts)

            # Create and load model
            from .models.psp import pSp
            self.net = pSp(self._opts)
            self.net.eval()
            if self.device == "cuda":
                self.net.cuda()

            self._loaded = True
            logger.info("SAM model loaded successfully on %s", self.device)
            return True

        except Exception as e:
            logger.error("Failed to load SAM model: %s", e)
            self._loaded = False
            return False

    @property
    def is_ready(self) -> bool:
        return self._loaded

    @torch.no_grad()
    def transform_face(self, face_crop_bgr: np.ndarray, target_age: int) -> np.ndarray:
        """
        Transform a pre-cropped face to the target age.

        Args:
            face_crop_bgr: BGR face crop (any size, will be resized to 256x256).
            target_age: Target age (0-100).

        Returns:
            Aged face crop as BGR numpy array (256x256).
        """
        if not self._loaded:
            raise RuntimeError("SAM model not loaded. Call load() first.")

        target_age = max(0, min(100, target_age))

        # Convert BGR -> RGB -> PIL
        face_rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
        face_pil = Image.fromarray(face_rgb)

        # Apply SAM transform: Resize, ToTensor, Normalize to [-1, 1]
        input_tensor = SAM_TRANSFORM(face_pil)

        # Add age channel via AgeTransformer
        from .datasets.augmentations import AgeTransformer
        age_transformer = AgeTransformer(target_age=target_age)
        input_with_age = age_transformer(input_tensor)  # 4-channel tensor

        # Batch dimension
        input_batch = input_with_age.unsqueeze(0).to(self.device).float()

        # Forward pass
        result_tensor = self.net(input_batch, randomize_noise=False, resize=True)

        # Convert tensor to PIL then to BGR numpy
        from .utils.common import tensor2im
        result_image = tensor2im(result_tensor[0])

        # PIL -> BGR numpy
        result_np = np.array(result_image)
        result_bgr = cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR)

        return result_bgr

    def align_face(self, image_path: str):
        """
        FFHQ-style face alignment using dlib 68-landmark predictor.

        Args:
            image_path: Path to the input image file.

        Returns:
            Aligned PIL Image (256x256) or None on failure.
        """
        try:
            import dlib
            if self._dlib_predictor is None:
                from .configs.paths_config import model_paths
                predictor_path = model_paths.get('shape_predictor', '')
                if not os.path.isfile(predictor_path):
                    logger.warning("dlib shape predictor not found at %s", predictor_path)
                    return None
                self._dlib_predictor = dlib.shape_predictor(predictor_path)

            from .scripts.align_face import align_face as _align
            aligned = _align(filepath=image_path, predictor=self._dlib_predictor)
            return aligned

        except ImportError:
            logger.warning("dlib not installed, cannot perform FFHQ alignment")
            return None
        except Exception as e:
            logger.warning("Face alignment failed: %s", e)
            return None
