"""
FADING (Face Aging via Diffusion-based Editing) Inference Wrapper
=================================================================
Fine-tuned Stable Diffusion for age-aware face editing using
null-text inversion + prompt-to-prompt attention control.

Supports bidirectional aging with target-age control.
Input:  face image (BGR numpy) + target age + current age + gender.
Output: aged face image (BGR numpy, 512x512).

Two execution modes:
  - Local GPU: loads the full pipeline on CUDA (requires ~6-8GB VRAM).
  - Modal Cloud: sends the image to a Modal serverless GPU endpoint
    when no local GPU is available. Set FADING_MODAL_ENDPOINT env var
    or Django setting to enable.

Reference: MunchkinChen/FADING (BMVC 2023).
"""

import base64
import logging
import os
import tempfile

import cv2
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger("agevision.diffusion_aging")

# Default checkpoint path
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))  # agevision_backend/
DEFAULT_CHECKPOINT_DIR = os.path.join(_BASE_DIR, 'checkpoints', 'fading')

# Google Drive file ID for FADING pretrained weights
FADING_GDOWN_ID = '1galwrcHq1HoZNfOI4jdJJqVs5ehB_dvO'


def _get_modal_endpoint_url() -> str | None:
    """Get the Modal FADING endpoint URL from settings or env."""
    # 1. Environment variable (highest priority)
    url = os.environ.get('FADING_MODAL_ENDPOINT')
    if url:
        return url.rstrip('/')

    # 2. Django settings
    try:
        from django.conf import settings
        url = getattr(settings, 'FADING_MODAL_ENDPOINT', None)
        if url:
            return url.rstrip('/')
    except Exception:
        pass

    return None


class ModalFADINGClient:
    """
    Calls the FADING Modal serverless GPU endpoint over HTTP.

    Same interface as DiffusionAgingInference so it can be used
    as a drop-in replacement when no local GPU is available.
    """

    def __init__(self, endpoint_url: str):
        self.endpoint_url = endpoint_url
        self._ready = True

    def load(self) -> bool:
        return True

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def device(self) -> str:
        return "modal-cloud-gpu"

    def transform_face(self, face_crop_bgr: np.ndarray, target_age: int,
                       current_age: int = 25, gender: str = 'unknown') -> np.ndarray:
        """Send the face to Modal cloud GPU for FADING inference."""
        import requests

        # Encode image as base64 JPEG
        _, buf = cv2.imencode('.jpg', face_crop_bgr,
                              [cv2.IMWRITE_JPEG_QUALITY, 95])
        image_b64 = base64.b64encode(buf.tobytes()).decode('utf-8')

        payload = {
            "image_b64": image_b64,
            "current_age": current_age,
            "target_age": target_age,
            "gender": gender,
        }

        logger.info("Sending face to Modal cloud GPU for FADING inference...")
        resp = requests.post(self.endpoint_url, json=payload, timeout=600)
        resp.raise_for_status()

        result = resp.json()
        if not result.get("success"):
            raise RuntimeError(
                f"Modal FADING failed: {result.get('error', 'unknown error')}")

        # Decode result image
        result_bytes = base64.b64decode(result["image_b64"])
        result_array = np.frombuffer(result_bytes, dtype=np.uint8)
        output_bgr = cv2.imdecode(result_array, cv2.IMREAD_COLOR)

        if output_bgr is None:
            raise RuntimeError("Failed to decode FADING result from Modal")

        logger.info("FADING cloud inference complete.")
        return output_bgr


def _get_person_placeholder(age, gender='unknown'):
    """Return age/gender-appropriate text placeholder for prompts."""
    is_female = gender.lower() in ('female', 'woman', 'girl')
    is_male = gender.lower() in ('male', 'man', 'boy')

    if age is not None and age <= 15:
        if is_female:
            return 'girl'
        elif is_male:
            return 'boy'
        return 'child'
    else:
        if is_female:
            return 'woman'
        elif is_male:
            return 'man'
        return 'person'


class DiffusionAgingInference:
    """
    FADING inference wrapper.

    Interface (matches FastAgingInference / SAMInference pattern):
      - load() -> bool
      - is_ready -> bool
      - transform_face(face_crop_bgr, target_age, current_age, gender) -> np.ndarray
    """

    def __init__(self, checkpoint_dir: str = None, device: str = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.checkpoint_dir = checkpoint_dir or DEFAULT_CHECKPOINT_DIR
        self._loaded = False
        self.pipe = None
        self.tokenizer = None

    def _download_checkpoint(self) -> bool:
        """Download FADING weights via gdown if not present."""
        try:
            import gdown
        except ImportError:
            logger.error("gdown is required to download FADING weights. "
                         "Install with: pip install gdown")
            return False

        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Check if already extracted (look for model_index.json which
        # indicates a valid diffusers model directory)
        if os.path.isfile(os.path.join(self.checkpoint_dir, 'model_index.json')):
            return True

        zip_path = os.path.join(self.checkpoint_dir, 'fading_weights.zip')
        try:
            logger.info("Downloading FADING weights from Google Drive...")
            gdown.download(id=FADING_GDOWN_ID, output=zip_path, quiet=False)

            if not os.path.isfile(zip_path):
                logger.error("gdown download failed — file not created")
                return False

            logger.info("Extracting FADING weights...")
            import zipfile
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(self.checkpoint_dir)

            # Clean up zip
            os.remove(zip_path)

            # The zip extracts to a subdirectory — find it and use it
            # as the checkpoint path
            subdirs = [d for d in os.listdir(self.checkpoint_dir)
                       if os.path.isdir(os.path.join(self.checkpoint_dir, d))]
            for subdir in subdirs:
                candidate = os.path.join(self.checkpoint_dir, subdir)
                if os.path.isfile(os.path.join(candidate, 'model_index.json')):
                    self.checkpoint_dir = candidate
                    break

            logger.info("FADING weights extracted to %s", self.checkpoint_dir)
            return True

        except Exception as e:
            logger.error("Failed to download FADING weights: %s", e)
            if os.path.isfile(zip_path):
                os.remove(zip_path)
            return False

    def load(self) -> bool:
        """Load FADING pipeline. Auto-downloads if needed. Returns True on success."""
        if self._loaded:
            return True

        # Check for model_index.json (standard diffusers checkpoint marker)
        if not os.path.isfile(os.path.join(self.checkpoint_dir, 'model_index.json')):
            # Try subdirectories (zip may have extracted into a subfolder)
            found = False
            if os.path.isdir(self.checkpoint_dir):
                for entry in os.listdir(self.checkpoint_dir):
                    candidate = os.path.join(self.checkpoint_dir, entry)
                    if (os.path.isdir(candidate) and
                            os.path.isfile(os.path.join(candidate,
                                                        'model_index.json'))):
                        self.checkpoint_dir = candidate
                        found = True
                        break

            if not found:
                logger.info("FADING checkpoint not found, attempting download...")
                if not self._download_checkpoint():
                    return False

        try:
            from diffusers import StableDiffusionPipeline, DDIMScheduler

            logger.info("Loading FADING pipeline from %s ...",
                        self.checkpoint_dir)

            scheduler = DDIMScheduler(
                beta_start=0.00085, beta_end=0.012,
                beta_schedule="scaled_linear",
                clip_sample=False, set_alpha_to_one=False,
                steps_offset=1,
            )

            # Must load in float32 — null-text optimization requires gradients
            # through the UNet, which is incompatible with float16.
            self.pipe = StableDiffusionPipeline.from_pretrained(
                self.checkpoint_dir,
                scheduler=scheduler,
                safety_checker=None,
                torch_dtype=torch.float32,
            ).to(self.device)

            self.tokenizer = self.pipe.tokenizer
            self._loaded = True
            logger.info("FADING pipeline loaded on %s", self.device)
            return True

        except Exception as e:
            logger.error("Failed to load FADING pipeline: %s", e)
            self._loaded = False
            return False

    @property
    def is_ready(self) -> bool:
        return self._loaded

    def transform_face(self, face_crop_bgr: np.ndarray, target_age: int,
                       current_age: int = 25, gender: str = 'unknown') -> np.ndarray:
        """
        Age a face using FADING diffusion-based editing.

        Note: This method intentionally does NOT use @torch.no_grad() because
        null-text optimization requires gradient computation through the UNet.

        Args:
            face_crop_bgr: BGR face image (any size, will be resized to 512x512).
            target_age:    Target age (1-100).
            current_age:   Detected current age of the face.
            gender:        'male', 'female', or 'unknown'.

        Returns:
            Aged face as BGR numpy array (512x512).
        """
        if not self._loaded:
            raise RuntimeError("FADING not loaded. Call load() first.")

        from .null_inversion import NullInversion
        from .p2p import make_controller, p2p_text2image

        # Resize input to 512x512 and save to temp file for NullInversion
        face_rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
        face_pil = Image.fromarray(face_rgb).resize((512, 512),
                                                     Image.LANCZOS)

        # NullInversion expects a file path — write to temp
        tmp_fd, tmp_path = tempfile.mkstemp(suffix='.png')
        try:
            os.close(tmp_fd)
            face_pil.save(tmp_path)

            # Build inversion prompt
            person_ph = _get_person_placeholder(current_age, gender)
            inversion_prompt = (
                f"photo of {current_age} year old {person_ph}"
            )

            # Step 1: Null-text inversion (requires gradients)
            null_inversion = NullInversion(self.pipe)
            (image_gt, image_enc), x_t, uncond_embeddings = (
                null_inversion.invert(tmp_path, inversion_prompt,
                                      offsets=(0, 0, 0, 0), verbose=True)
            )

            # Step 2: Age editing via prompt-to-prompt (no gradients needed)
            new_person_ph = _get_person_placeholder(target_age, gender)
            new_prompt = (
                f"photo of {target_age} year old {new_person_ph}"
            )

            prompts = [inversion_prompt, new_prompt]

            blend_word = (
                (str(current_age), person_ph),
                (str(target_age), new_person_ph),
            )
            cross_replace_steps = {'default_': .8}
            self_replace_steps = .5
            eq_params = {"words": (str(target_age),),
                         "values": (1,)}

            controller = make_controller(
                prompts, True, cross_replace_steps,
                self_replace_steps, self.tokenizer,
                blend_word, eq_params
            )

            g_device = torch.device(self.device)
            g_cuda = torch.Generator(device=g_device)

            images, _ = p2p_text2image(
                self.pipe, prompts, controller,
                generator=g_cuda.manual_seed(0),
                latent=x_t, uncond_embeddings=uncond_embeddings
            )

            # Take the edited image (last in batch)
            output_rgb = images[-1]  # numpy uint8 H x W x 3 (RGB)

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        # RGB -> BGR for consistency with the rest of the pipeline
        output_bgr = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR)
        return output_bgr
