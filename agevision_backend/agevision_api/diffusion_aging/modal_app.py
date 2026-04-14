"""
Modal Cloud GPU Deployment for FADING
======================================
Deploy once:   modal deploy modal_app.py
Test locally:  modal run modal_app.py

This creates a serverless GPU endpoint that runs FADING inference.
Free tier: 30 GPU-hrs/month. Uses A10G for ~3min inference.
"""

import modal
import io
import base64

# ---------------------------------------------------------------------------
# Modal App & Image
# ---------------------------------------------------------------------------

app = modal.App("agevision-fading")

GDOWN_ID = "1galwrcHq1HoZNfOI4jdJJqVs5ehB_dvO"
CHECKPOINT_PATH = "/checkpoints/fading"

fading_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "diffusers>=0.25.0",
        "transformers>=4.30.0",
        "accelerate>=0.20.0",
        "numpy>=1.24.0",
        "Pillow>=10.0.0",
        "opencv-python-headless>=4.8.0",
        "tqdm>=4.65.0",
        "gdown>=5.0.0",
        "fastapi[standard]",
    )
)

# ---------------------------------------------------------------------------
# Volume for caching the FADING checkpoint (~2GB)
# ---------------------------------------------------------------------------

volume = modal.Volume.from_name("fading-checkpoints", create_if_missing=True)


# ---------------------------------------------------------------------------
# FADING Inference Class (runs on GPU)
# ---------------------------------------------------------------------------

@app.cls(
    gpu="A10G",
    image=fading_image,
    volumes={"/checkpoints": volume},
    timeout=600,
    scaledown_window=300,  # keep warm for 5 min between requests
)
class FADINGModel:
    """Serverless FADING inference on Modal GPU."""

    @modal.enter()
    def setup(self):
        """Download checkpoint if needed and load the FADING pipeline."""
        import os
        import subprocess
        import torch
        from diffusers import StableDiffusionPipeline, DDIMScheduler

        checkpoint_dir = CHECKPOINT_PATH
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Check if checkpoint already exists in the volume
        model_index_found = os.path.isfile(
            os.path.join(checkpoint_dir, 'model_index.json')
        )
        if not model_index_found and os.path.isdir(checkpoint_dir):
            for entry in os.listdir(checkpoint_dir):
                candidate = os.path.join(checkpoint_dir, entry)
                if (os.path.isdir(candidate) and
                        os.path.isfile(os.path.join(candidate,
                                                    'model_index.json'))):
                    checkpoint_dir = candidate
                    model_index_found = True
                    break

        if not model_index_found:
            print("Downloading FADING checkpoint...")
            zip_path = os.path.join(CHECKPOINT_PATH, "fading_weights.zip")
            # gdown v6+ uses positional URL, not --id flag
            gdown_url = f"https://drive.google.com/uc?id={GDOWN_ID}"
            subprocess.run([
                "gdown", "-O", zip_path, gdown_url
            ], check=True)

            import zipfile
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(CHECKPOINT_PATH)
            os.remove(zip_path)

            # Find the extracted checkpoint dir
            for entry in os.listdir(CHECKPOINT_PATH):
                candidate = os.path.join(CHECKPOINT_PATH, entry)
                if (os.path.isdir(candidate) and
                        os.path.isfile(os.path.join(candidate,
                                                    'model_index.json'))):
                    checkpoint_dir = candidate
                    break

            volume.commit()
            print("FADING checkpoint downloaded and committed to volume.")

        # Load the pipeline
        scheduler = DDIMScheduler(
            beta_start=0.00085, beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False, set_alpha_to_one=False,
            steps_offset=1,
        )

        self.device = "cuda"
        self.pipe = StableDiffusionPipeline.from_pretrained(
            checkpoint_dir,
            scheduler=scheduler,
            safety_checker=None,
            torch_dtype=torch.float32,
        ).to(self.device)

        self.tokenizer = self.pipe.tokenizer
        print(f"FADING pipeline loaded on {self.device}")

    @modal.method()
    def transform(self, image_b64: str, current_age: int,
                  target_age: int, gender: str = "unknown") -> dict:
        """
        Run FADING age transformation.

        Args:
            image_b64: Base64-encoded face image (PNG/JPEG).
            current_age: Detected current age.
            target_age: Target age (1-100).
            gender: 'male', 'female', or 'unknown'.

        Returns:
            dict with 'image_b64' (base64-encoded result) and 'success'.
        """
        import os
        import tempfile
        import cv2
        import numpy as np
        import torch
        from PIL import Image

        try:
            # Decode input image
            img_bytes = base64.b64decode(image_b64)
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            if img_bgr is None:
                return {"success": False, "error": "Failed to decode image"}

            # Resize to 512x512
            face_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            face_pil = Image.fromarray(face_rgb).resize((512, 512),
                                                         Image.LANCZOS)

            # Write to temp file for NullInversion
            tmp_fd, tmp_path = tempfile.mkstemp(suffix='.png')
            os.close(tmp_fd)
            face_pil.save(tmp_path)

            try:
                result_bgr = self._run_fading(tmp_path, current_age,
                                              target_age, gender)
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

            # Encode result as base64 JPEG
            _, buf = cv2.imencode('.jpg', result_bgr,
                                  [cv2.IMWRITE_JPEG_QUALITY, 95])
            result_b64 = base64.b64encode(buf.tobytes()).decode('utf-8')

            return {"success": True, "image_b64": result_b64}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _run_fading(self, image_path, current_age, target_age, gender):
        """Core FADING logic (null-text inversion + p2p editing)."""
        import numpy as np
        import torch
        import cv2
        from typing import Union, Optional, Tuple, List, Dict
        from tqdm import tqdm
        import torch.nn.functional as nnf
        from PIL import Image
        from torch.optim.adam import Adam
        import abc

        NUM_DDIM_STEPS = 25
        GUIDANCE_SCALE = 7.5
        MAX_NUM_WORDS = 77
        device = torch.device(self.device)

        # ── Person placeholder ──
        def get_person_placeholder(age, g='unknown'):
            is_female = g.lower() in ('female', 'woman', 'girl')
            is_male = g.lower() in ('male', 'man', 'boy')
            if age and age <= 15:
                if is_female: return 'girl'
                if is_male: return 'boy'
                return 'child'
            if is_female: return 'woman'
            if is_male: return 'man'
            return 'person'

        # ── Load image 512x512 ──
        def load_512(path, left=0, right=0, top=0, bottom=0):
            image = np.array(Image.open(path))[:, :, :3]
            h, w, c = image.shape
            left = min(left, w - 1)
            right = min(right, w - left - 1)
            top = min(top, h - 1)
            bottom = min(bottom, h - top - 1)
            image = image[top:h - bottom, left:w - right]
            h, w, c = image.shape
            if h < w:
                offset = (w - h) // 2
                image = image[:, offset:offset + h]
            elif w < h:
                offset = (h - w) // 2
                image = image[offset:offset + w]
            return np.array(Image.fromarray(image).resize((512, 512)))

        # ── ptp_utils (inline) ──
        def diffusion_step(model, controller, latents, context, t,
                           guidance_scale, low_resource=False):
            latents_input = torch.cat([latents] * 2)
            noise_pred = model.unet(latents_input, t,
                                    encoder_hidden_states=context)["sample"]
            noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * \
                         (noise_prediction_text - noise_pred_uncond)
            latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
            latents = controller.step_callback(latents)
            return latents

        def latent2image(vae, latents):
            latents = 1 / 0.18215 * latents
            image = vae.decode(latents)['sample']
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()
            return (image * 255).astype(np.uint8)

        def init_latent(latent, model, height, width, generator, batch_size):
            if latent is None:
                latent = torch.randn(
                    (1, model.unet.config.in_channels, height // 8, width // 8),
                    generator=generator, device=model.device)
            latents = latent.expand(
                batch_size, model.unet.config.in_channels,
                height // 8, width // 8).to(model.device)
            return latent, latents

        def get_word_inds(text, word_place, tokenizer):
            split_text = text.split(" ")
            if type(word_place) is str:
                word_place = [i for i, w in enumerate(split_text)
                              if word_place == w]
            elif type(word_place) is int:
                word_place = [word_place]
            out = []
            if len(word_place) > 0:
                words_encode = [tokenizer.decode([item]).strip("#")
                                for item in tokenizer.encode(text)][1:-1]
                cur_len, ptr = 0, 0
                for i in range(len(words_encode)):
                    cur_len += len(words_encode[i])
                    if ptr in word_place:
                        out.append(i + 1)
                    if cur_len >= len(split_text[ptr]):
                        ptr += 1
                        cur_len = 0
            return np.array(out)

        def update_alpha_time_word(alpha, bounds, prompt_ind, word_inds=None):
            if type(bounds) is float:
                bounds = 0, bounds
            start = int(bounds[0] * alpha.shape[0])
            end = int(bounds[1] * alpha.shape[0])
            if word_inds is None:
                word_inds = torch.arange(alpha.shape[2])
            alpha[: start, prompt_ind, word_inds] = 0
            alpha[start: end, prompt_ind, word_inds] = 1
            alpha[end:, prompt_ind, word_inds] = 0
            return alpha

        def get_time_words_attention_alpha(prompts, num_steps,
                                           cross_replace_steps, tokenizer,
                                           max_num_words=77):
            if type(cross_replace_steps) is not dict:
                cross_replace_steps = {"default_": cross_replace_steps}
            if "default_" not in cross_replace_steps:
                cross_replace_steps["default_"] = (0., 1.)
            alpha = torch.zeros(num_steps + 1, len(prompts) - 1, max_num_words)
            for i in range(len(prompts) - 1):
                alpha = update_alpha_time_word(
                    alpha, cross_replace_steps["default_"], i)
            for key, item in cross_replace_steps.items():
                if key != "default_":
                    inds = [get_word_inds(prompts[i], key, tokenizer)
                            for i in range(1, len(prompts))]
                    for i, ind in enumerate(inds):
                        if len(ind) > 0:
                            alpha = update_alpha_time_word(alpha, item, i, ind)
            return alpha.reshape(
                num_steps + 1, len(prompts) - 1, 1, 1, max_num_words)

        def register_attention_control(model, controller):
            def ca_forward(self_attn, place_in_unet):
                to_out = self_attn.to_out
                if type(to_out) is torch.nn.modules.container.ModuleList:
                    to_out = self_attn.to_out[0]

                def forward(hidden_states, encoder_hidden_states=None,
                            attention_mask=None, **kwargs):
                    x = hidden_states
                    context = encoder_hidden_states
                    batch_size, seq_len, dim = x.shape
                    h = self_attn.heads
                    q = self_attn.to_q(x)
                    is_cross = context is not None
                    context = context if is_cross else x
                    k = self_attn.to_k(context)
                    v = self_attn.to_v(context)
                    q = self_attn.head_to_batch_dim(q)
                    k = self_attn.head_to_batch_dim(k)
                    v = self_attn.head_to_batch_dim(v)
                    sim = torch.einsum("b i d, b j d -> b i j", q, k) * self_attn.scale
                    attn = sim.softmax(dim=-1)
                    attn = ctrl(attn, is_cross, place_in_unet)
                    out = torch.einsum("b i j, b j d -> b i d", attn, v)
                    out = self_attn.batch_to_head_dim(out)
                    return to_out(out)
                return forward

            class DummyController:
                def __call__(self, *args): return args[0]
                def __init__(self): self.num_att_layers = 0

            ctrl = controller if controller is not None else DummyController()

            def register_recr(net_, count, place):
                cls_name = net_.__class__.__name__
                if cls_name in ('CrossAttention', 'Attention') and hasattr(net_, 'to_q'):
                    net_.forward = ca_forward(net_, place)
                    return count + 1
                elif hasattr(net_, 'children'):
                    for child in net_.children():
                        count = register_recr(child, count, place)
                return count

            cross_att_count = 0
            for name, net in model.unet.named_children():
                if "down" in name:
                    cross_att_count += register_recr(net, 0, "down")
                elif "up" in name:
                    cross_att_count += register_recr(net, 0, "up")
                elif "mid" in name:
                    cross_att_count += register_recr(net, 0, "mid")
            ctrl.num_att_layers = cross_att_count

        # ── seq_aligner (inline) ──
        def get_replacement_mapper_(x, y, tokenizer, max_len=77):
            words_x, words_y = x.split(' '), y.split(' ')
            if len(words_x) != len(words_y):
                raise ValueError("Prompts must have same word count for replacement")
            inds_replace = [i for i in range(len(words_y)) if words_y[i] != words_x[i]]
            inds_source = [get_word_inds(x, i, tokenizer) for i in inds_replace]
            inds_target = [get_word_inds(y, i, tokenizer) for i in inds_replace]
            mapper = np.zeros((max_len, max_len))
            i = j = cur = 0
            while i < max_len and j < max_len:
                if cur < len(inds_source) and len(inds_source[cur]) > 0 and inds_source[cur][0] == i:
                    s, t_ = inds_source[cur], inds_target[cur]
                    if len(s) == len(t_):
                        mapper[s, t_] = 1
                    else:
                        ratio = 1 / len(t_)
                        for it in t_:
                            mapper[s, it] = ratio
                    cur += 1
                    i += len(s)
                    j += len(t_)
                else:
                    mapper[i, j] = 1
                    i += 1
                    j += 1
            return torch.from_numpy(mapper).float()

        def get_replacement_mapper(prompts, tokenizer, max_len=77):
            mappers = []
            for i in range(1, len(prompts)):
                mappers.append(
                    get_replacement_mapper_(prompts[0], prompts[i],
                                            tokenizer, max_len))
            return torch.stack(mappers)

        # ── Attention Controllers ──
        class EmptyControl:
            def step_callback(self, x_t): return x_t
            def between_steps(self): return
            def __call__(self, attn, is_cross, place): return attn

        class AttentionStore(abc.ABC):
            @staticmethod
            def get_empty_store():
                return {"down_cross": [], "mid_cross": [], "up_cross": [],
                        "down_self": [], "mid_self": [], "up_self": []}

            def forward(self, attn, is_cross, place):
                key = f"{place}_{'cross' if is_cross else 'self'}"
                if attn.shape[1] <= 32 ** 2:
                    self.step_store[key].append(attn)
                return attn

            def between_steps(self):
                if len(self.attention_store) == 0:
                    self.attention_store = self.step_store
                else:
                    for key in self.attention_store:
                        for i in range(len(self.attention_store[key])):
                            self.attention_store[key][i] += self.step_store[key][i]
                self.step_store = self.get_empty_store()

            def step_callback(self, x_t): return x_t

            def __call__(self, attn, is_cross, place):
                if self.cur_att_layer >= 0:
                    h = attn.shape[0]
                    attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place)
                    self.cur_att_layer += 1
                    if self.cur_att_layer == self.num_att_layers:
                        self.cur_att_layer = 0
                        self.cur_step += 1
                        self.between_steps()
                return attn

            def reset(self):
                self.cur_step = 0
                self.cur_att_layer = 0
                self.step_store = self.get_empty_store()
                self.attention_store = {}

            def __init__(self):
                self.cur_step = 0
                self.num_att_layers = -1
                self.cur_att_layer = 0
                self.step_store = self.get_empty_store()
                self.attention_store = {}

        class AttentionReplace(AttentionStore):
            def step_callback(self, x_t):
                if self.local_blend is not None:
                    x_t = self.local_blend(x_t, self.attention_store)
                return x_t

            def forward(self, attn, is_cross, place):
                super().forward(attn, is_cross, place)
                if is_cross or (self.num_self_replace[0] <= self.cur_step <
                                self.num_self_replace[1]):
                    h = attn.shape[0] // self.batch_size
                    attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
                    attn_base, attn_replace = attn[0], attn[1:]
                    if is_cross:
                        alpha_words = self.cross_replace_alpha[self.cur_step]
                        attn_new = torch.einsum(
                            'hpw,bwn->bhpn', attn_base, self.mapper)
                        attn_new = attn_new * alpha_words + (1 - alpha_words) * attn_replace
                        attn[1:] = attn_new
                    else:
                        if attn_replace.shape[2] <= 32 ** 2:
                            attn_base_exp = attn_base.unsqueeze(0).expand(
                                attn_replace.shape[0], *attn_base.shape)
                            attn[1:] = attn_base_exp
                    attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
                return attn

            def __init__(self, prompts, num_steps, cross_replace_steps,
                         self_replace_steps, tokenizer, local_blend=None):
                super().__init__()
                self.batch_size = len(prompts)
                self.cross_replace_alpha = get_time_words_attention_alpha(
                    prompts, num_steps, cross_replace_steps, tokenizer
                ).to(device)
                if type(self_replace_steps) is float:
                    self_replace_steps = 0, self_replace_steps
                self.num_self_replace = (
                    int(num_steps * self_replace_steps[0]),
                    int(num_steps * self_replace_steps[1]))
                self.local_blend = local_blend
                self.mapper = get_replacement_mapper(
                    prompts, tokenizer).to(device)

        class AttentionReweight(AttentionReplace):
            def forward(self, attn, is_cross, place):
                attn = super().forward(attn, is_cross, place)
                if is_cross:
                    h = attn.shape[0] // self.batch_size
                    attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
                    attn[1:] = attn[1:] * self.equalizer[:, None, None, :]
                    attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
                return attn

            def __init__(self, prompts, num_steps, cross_replace_steps,
                         self_replace_steps, tokenizer, equalizer,
                         local_blend=None):
                super().__init__(prompts, num_steps, cross_replace_steps,
                                 self_replace_steps, tokenizer, local_blend)
                self.equalizer = equalizer.to(device)

        # ── NullInversion ──
        class NullInversion:
            def __init__(self, model):
                self.model = model
                self.tokenizer = model.tokenizer
                self.model.scheduler.set_timesteps(NUM_DDIM_STEPS)
                self.prompt = None
                self.context = None

            @property
            def scheduler(self):
                return self.model.scheduler

            def prev_step(self, model_output, timestep, sample):
                prev_ts = (timestep -
                           self.scheduler.config.num_train_timesteps //
                           self.scheduler.num_inference_steps)
                alpha_t = self.scheduler.alphas_cumprod[timestep]
                alpha_prev = (self.scheduler.alphas_cumprod[prev_ts]
                              if prev_ts >= 0
                              else self.scheduler.final_alpha_cumprod)
                beta_t = 1 - alpha_t
                pred = (sample - beta_t ** 0.5 * model_output) / alpha_t ** 0.5
                direction = (1 - alpha_prev) ** 0.5 * model_output
                return alpha_prev ** 0.5 * pred + direction

            def next_step(self, model_output, timestep, sample):
                ts = min(timestep - self.scheduler.config.num_train_timesteps //
                         self.scheduler.num_inference_steps, 999)
                next_ts = timestep
                alpha_t = (self.scheduler.alphas_cumprod[ts] if ts >= 0
                           else self.scheduler.final_alpha_cumprod)
                alpha_next = self.scheduler.alphas_cumprod[next_ts]
                beta_t = 1 - alpha_t
                pred = (sample - beta_t ** 0.5 * model_output) / alpha_t ** 0.5
                direction = (1 - alpha_next) ** 0.5 * model_output
                return alpha_next ** 0.5 * pred + direction

            def get_noise_pred_single(self, latents, t, context):
                return self.model.unet(
                    latents, t, encoder_hidden_states=context)["sample"]

            @torch.no_grad()
            def latent2image(self, latents):
                latents = 1 / 0.18215 * latents.detach()
                image = self.model.vae.decode(latents)['sample']
                image = (image / 2 + 0.5).clamp(0, 1)
                image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
                return (image * 255).astype(np.uint8)

            @torch.no_grad()
            def image2latent(self, image):
                if type(image) is torch.Tensor and image.dim() == 4:
                    return image
                image = torch.from_numpy(image).float() / 127.5 - 1
                image = image.permute(2, 0, 1).unsqueeze(0).to(device)
                latents = self.model.vae.encode(image)['latent_dist'].mean
                return latents * 0.18215

            @torch.no_grad()
            def init_prompt(self, prompt):
                uncond = self.tokenizer(
                    [""], padding="max_length",
                    max_length=self.tokenizer.model_max_length,
                    return_tensors="pt")
                uncond_emb = self.model.text_encoder(
                    uncond.input_ids.to(device))[0]
                text = self.tokenizer(
                    [prompt], padding="max_length",
                    max_length=self.tokenizer.model_max_length,
                    truncation=True, return_tensors="pt")
                text_emb = self.model.text_encoder(
                    text.input_ids.to(device))[0]
                self.context = torch.cat([uncond_emb, text_emb])
                self.prompt = prompt

            @torch.no_grad()
            def ddim_loop(self, latent):
                _, cond = self.context.chunk(2)
                all_latent = [latent]
                latent = latent.clone().detach()
                for i in range(NUM_DDIM_STEPS):
                    t = self.scheduler.timesteps[
                        len(self.scheduler.timesteps) - i - 1]
                    noise = self.get_noise_pred_single(latent, t, cond)
                    latent = self.next_step(noise, t, latent)
                    all_latent.append(latent)
                return all_latent

            @torch.no_grad()
            def ddim_inversion(self, image):
                latent = self.image2latent(image)
                image_rec = self.latent2image(latent)
                return image_rec, self.ddim_loop(latent)

            def null_optimization(self, latents, num_inner_steps, epsilon):
                uncond, cond = self.context.chunk(2)
                uncond_list = []
                latent_cur = latents[-1]
                bar = tqdm(total=num_inner_steps * NUM_DDIM_STEPS,
                           desc="Null-text opt")
                for i in range(NUM_DDIM_STEPS):
                    uncond = uncond.clone().detach()
                    uncond.requires_grad = True
                    optimizer = Adam([uncond], lr=1e-2 * (1. - i / 100.))
                    latent_prev = latents[len(latents) - i - 2]
                    t = self.scheduler.timesteps[i]
                    with torch.no_grad():
                        noise_cond = self.get_noise_pred_single(
                            latent_cur, t, cond)
                    for j in range(num_inner_steps):
                        noise_uncond = self.get_noise_pred_single(
                            latent_cur, t, uncond)
                        noise_pred = noise_uncond + GUIDANCE_SCALE * (
                            noise_cond - noise_uncond)
                        lat_prev_rec = self.prev_step(noise_pred, t, latent_cur)
                        loss = nnf.mse_loss(lat_prev_rec, latent_prev)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        bar.update()
                        if loss.item() < epsilon + i * 2e-5:
                            break
                    for j in range(j + 1, num_inner_steps):
                        bar.update()
                    uncond_list.append(uncond[:1].detach())
                    with torch.no_grad():
                        ctx = torch.cat([uncond, cond])
                        latents_input = torch.cat([latent_cur] * 2)
                        noise = self.model.unet(
                            latents_input, t,
                            encoder_hidden_states=ctx)["sample"]
                        nu, nt = noise.chunk(2)
                        noise = nu + GUIDANCE_SCALE * (nt - nu)
                        latent_cur = self.prev_step(noise, t, latent_cur)
                bar.close()
                return uncond_list

            def invert(self, image_path, prompt, offsets=(0, 0, 0, 0),
                       num_inner_steps=5, early_stop_epsilon=1e-5):
                self.init_prompt(prompt)
                register_attention_control(self.model, None)
                image_gt = load_512(image_path, *offsets)
                print("DDIM inversion...")
                image_rec, ddim_latents = self.ddim_inversion(image_gt)
                print("Null-text optimization...")
                uncond = self.null_optimization(
                    ddim_latents, num_inner_steps, early_stop_epsilon)
                return (image_gt, image_rec), ddim_latents[-1], uncond

        # ── p2p_text2image ──
        @torch.no_grad()
        def p2p_text2image(model, prompt, controller, num_inference_steps=NUM_DDIM_STEPS,
                           guidance_scale=7.5, generator=None, latent=None,
                           uncond_embeddings=None, start_time=NUM_DDIM_STEPS):
            batch_size = len(prompt)
            register_attention_control(model, controller)
            text_input = model.tokenizer(
                prompt, padding="max_length",
                max_length=model.tokenizer.model_max_length,
                truncation=True, return_tensors="pt")
            text_emb = model.text_encoder(
                text_input.input_ids.to(device))[0]
            uncond_ = None
            if uncond_embeddings is None:
                uncond_input = model.tokenizer(
                    [""] * batch_size, padding="max_length",
                    max_length=text_input.input_ids.shape[-1],
                    return_tensors="pt")
                uncond_ = model.text_encoder(
                    uncond_input.input_ids.to(device))[0]

            latent, latents = init_latent(latent, model, 512, 512,
                                          generator, batch_size)
            model.scheduler.set_timesteps(num_inference_steps)

            for i, t in enumerate(tqdm(
                    model.scheduler.timesteps[-start_time:],
                    desc="Age editing")):
                if uncond_ is None:
                    context = torch.cat([
                        uncond_embeddings[i].expand(*text_emb.shape),
                        text_emb])
                else:
                    context = torch.cat([uncond_, text_emb])
                latents = diffusion_step(model, controller, latents,
                                         context, t, guidance_scale)
            return latent2image(model.vae, latents), latent

        # ── RUN ──
        person_ph = get_person_placeholder(current_age, gender)
        inv_prompt = f"photo of {current_age} year old {person_ph}"

        null_inv = NullInversion(self.pipe)
        (_, _), x_t, uncond_embs = null_inv.invert(image_path, inv_prompt)

        new_ph = get_person_placeholder(target_age, gender)
        new_prompt = f"photo of {target_age} year old {new_ph}"
        prompts = [inv_prompt, new_prompt]

        eq_params_words = (str(target_age),)
        equalizer = torch.ones(1, 77)
        for w in eq_params_words:
            inds = get_word_inds(prompts[1], w, self.tokenizer)
            equalizer[:, inds] = 1

        controller = AttentionReplace(
            prompts, NUM_DDIM_STEPS,
            cross_replace_steps={'default_': .8},
            self_replace_steps=.5,
            tokenizer=self.tokenizer)

        g = torch.Generator(device=device).manual_seed(0)
        images, _ = p2p_text2image(
            self.pipe, prompts, controller,
            generator=g, latent=x_t, uncond_embeddings=uncond_embs)

        output_rgb = images[-1]
        return cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------------
# Web endpoint for HTTP access from Django backend
# ---------------------------------------------------------------------------

@app.function(image=fading_image, timeout=600)
@modal.fastapi_endpoint(method="POST")
def age_face_endpoint(data: dict) -> dict:
    """HTTP endpoint: POST {image_b64, current_age, target_age, gender}."""
    model = FADINGModel()
    return model.transform.remote(
        image_b64=data["image_b64"],
        current_age=data["current_age"],
        target_age=data["target_age"],
        gender=data.get("gender", "unknown"),
    )


# ---------------------------------------------------------------------------
# Local test entry point
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main():
    print("Testing FADING Modal deployment...")
    print("Deploy with: modal deploy modal_app.py")
    print("This will give you a URL like:")
    print("  https://your-user--agevision-fading-age-face-endpoint.modal.run")
