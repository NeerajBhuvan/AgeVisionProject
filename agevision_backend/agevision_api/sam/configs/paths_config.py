import os

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))  # agevision_backend/

model_paths = {
    'shape_predictor': os.path.join(_BASE_DIR, 'checkpoints', 'shape_predictor_68_face_landmarks.dat'),
    'sam_ffhq_aging': os.path.join(_BASE_DIR, 'checkpoints', 'sam_ffhq_aging.pt'),
    'sam_indian': os.path.join(_BASE_DIR, 'checkpoints', 'sam_indian_best.pt'),
    # Training-only paths (not required for inference from checkpoint)
    'ir_se50': os.path.join(_BASE_DIR, 'checkpoints', 'model_ir_se50.pth'),
    'stylegan_ffhq': os.path.join(_BASE_DIR, 'checkpoints', 'stylegan2-ffhq-config-f.pt'),
    'pretrained_psp': os.path.join(_BASE_DIR, 'checkpoints', 'psp_ffhq_encode.pt'),
    'pretrained_psp_encoder': os.path.join(_BASE_DIR, 'checkpoints', 'psp_ffhq_encode.pt'),
    'age_predictor': os.path.join(_BASE_DIR, 'checkpoints', 'dex_age_classifier.pth'),
    # MiVOLO v2 Indian-finetuned checkpoint (optional, auto-detected by mivolo_predictor)
    'mivolo_indian': os.path.join(_BASE_DIR, 'checkpoints', 'mivolo_indian_best.pt'),
}
