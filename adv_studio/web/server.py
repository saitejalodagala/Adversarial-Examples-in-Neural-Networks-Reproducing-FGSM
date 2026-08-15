import os
from typing import Dict, List, Optional
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from adv_studio.models import get_model, SimpleCNN
from adv_studio.attacks import get_attack, ATTACK_REGISTRY
from adv_studio.data.transforms import base64_png_to_tensor, tensor_to_base64_png
from adv_studio.data.mnist_loader import get_sample_digits, RawMNISTDataset
from adv_studio.defenses.preprocessing import (
    BitDepthReduction,
    SpatialSmoothing,
    TotalVariationDenoising,
)
from adv_studio.evaluation.metrics import compute_distortion_metrics
from adv_studio.evaluation.loss_landscape import compute_loss_landscape_1d

app = FastAPI(
    title="Adversarial Studio API",
    description="Backend for real-time adversarial attack generation, defense benchmarking, and neural network visualization.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Device & Model Cache
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
CHECKPOINTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "checkpoints")

MODELS_CACHE: Dict[str, nn.Module] = {}


def load_cached_model(name: str) -> nn.Module:
    """Load model from cache or instantiate with pretrained checkpoint."""
    if name in MODELS_CACHE:
        return MODELS_CACHE[name]

    ckpt_path = None
    if name == "clean_cnn" or name == "simple_cnn":
        ckpt_path = os.path.join(CHECKPOINTS_DIR, "mnist_cnn.pth")
    elif name == "adv_cnn" or name == "adv_trained_cnn":
        ckpt_path = os.path.join(CHECKPOINTS_DIR, "mnist_cnn_adv_trained.pth")

    model = get_model(name="simple_cnn", pretrained_path=ckpt_path, device=DEVICE)
    model.eval()
    MODELS_CACHE[name] = model
    return model


# Pydantic Request Schemas
class PredictRequest(BaseModel):
    image_base64: str
    model_name: str = "clean_cnn"


class AttackRequest(BaseModel):
    image_base64: str
    true_label: Optional[int] = None
    target_label: Optional[int] = None
    model_name: str = "clean_cnn"
    attack_name: str = "fgsm"
    epsilon: float = 0.2
    alpha: Optional[float] = None
    steps: int = 10
    targeted: bool = False


class DefenseRequest(BaseModel):
    image_base64: str
    defense_type: str = "bit_depth"  # 'bit_depth', 'spatial_blur', 'tv_denoise'
    model_name: str = "clean_cnn"
    param_value: float = 4.0


class LandscapeRequest(BaseModel):
    image_base64: str
    adv_image_base64: str
    true_label: int
    model_name: str = "clean_cnn"
    num_points: int = 40


# API Endpoints
@app.get("/api/models")
def get_models_list():
    return {
        "models": [
            {"id": "clean_cnn", "name": "SimpleCNN (Clean Trained)", "robust": False},
            {"id": "adv_cnn", "name": "SimpleCNN (Adversarially Trained, eps=0.2)", "robust": True},
        ]
    }


@app.get("/api/attacks")
def get_attacks_list():
    return {
        "attacks": [
            {"id": "fgsm", "name": "FGSM (Fast Gradient Sign Method)", "iterative": False, "supports_target": True},
            {"id": "ifgsm", "name": "I-FGSM (Basic Iterative Method)", "iterative": True, "supports_target": True},
            {"id": "pgd", "name": "PGD (Projected Gradient Descent, Linf)", "iterative": True, "supports_target": True},
            {"id": "cw", "name": "Carlini-Wagner (CW-L2 Optimization)", "iterative": True, "supports_target": True},
            {"id": "deepfool", "name": "DeepFool (Minimal L2 Perturbation)", "iterative": True, "supports_target": False},
            {"id": "mifgsm", "name": "MI-FGSM (Momentum Iterative FGSM)", "iterative": True, "supports_target": True},
            {"id": "random_noise", "name": "Random Noise (Uniform Baseline)", "iterative": False, "supports_target": False},
        ]
    }


@app.get("/api/defenses")
def get_defenses_list():
    return {
        "defenses": [
            {"id": "bit_depth", "name": "Bit-Depth Quantization", "param_name": "Bits (1-8)", "default_val": 3.0, "min_val": 1.0, "max_val": 8.0},
            {"id": "spatial_blur", "name": "Spatial Gaussian Smoothing", "param_name": "Sigma", "default_val": 1.0, "min_val": 0.2, "max_val": 3.0},
            {"id": "tv_denoise", "name": "Total Variation Denoising", "param_name": "TV Weight", "default_val": 0.05, "min_val": 0.01, "max_val": 0.2},
        ]
    }


@app.get("/api/samples")
def get_dataset_samples():
    """Return 10 sample digits (0-9) from test dataset."""
    try:
        sample_imgs, sample_lbls = get_sample_digits(count=10)
        samples = []
        for i in range(sample_imgs.size(0)):
            b64 = tensor_to_base64_png(sample_imgs[i])
            samples.append({
                "index": i,
                "label": int(sample_lbls[i].item()),
                "image_base64": b64,
            })
        return {"samples": samples}
    except Exception as e:
        # Fallback synthetic digits
        samples = []
        for i in range(10):
            t = torch.zeros((1, 28, 28))
            # Draw rough number
            t[0, 10:20, 10:20] = 0.8
            samples.append({
                "index": i,
                "label": i,
                "image_base64": tensor_to_base64_png(t),
            })
        return {"samples": samples}


@app.post("/api/predict")
def predict_image(req: PredictRequest):
    model = load_cached_model(req.model_name)
    tensor = base64_png_to_tensor(req.image_base64).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        pred = int(probs.argmax())
        conf = float(probs[pred])

    return {
        "prediction": pred,
        "confidence": conf,
        "probabilities": [float(p) for p in probs],
    }


@app.post("/api/attack")
def attack_image(req: AttackRequest):
    model = load_cached_model(req.model_name)
    tensor = base64_png_to_tensor(req.image_base64).to(DEVICE)

    # Determine original prediction if true_label not specified
    with torch.no_grad():
        orig_logits = model(tensor)
        orig_probs = torch.softmax(orig_logits, dim=1).squeeze(0).cpu().numpy()
        orig_pred = int(orig_probs.argmax())
        orig_conf = float(orig_probs[orig_pred])

    label_val = req.true_label if req.true_label is not None else orig_pred
    lbl_tensor = torch.tensor([label_val], dtype=torch.long, device=DEVICE)

    target_tensor = None
    if req.targeted and req.target_label is not None:
        target_tensor = torch.tensor([req.target_label], dtype=torch.long, device=DEVICE)

    # Instantiate attack
    attack = get_attack(
        name=req.attack_name,
        model=model,
        epsilon=req.epsilon,
        alpha=req.alpha,
        steps=req.steps,
        targeted=req.targeted,
        device=DEVICE,
    )

    adv_tensor = attack.generate(tensor, lbl_tensor, target_labels=target_tensor)

    # Evaluate adversarial image
    with torch.no_grad():
        adv_logits = model(adv_tensor)
        adv_probs = torch.softmax(adv_logits, dim=1).squeeze(0).cpu().numpy()
        adv_pred = int(adv_probs.argmax())
        adv_conf = float(adv_probs[adv_pred])

    # Compute perturbation difference map
    diff_tensor = adv_tensor - tensor
    # Scale difference map to [0, 1] for visualization: (diff + eps) / (2 * eps)
    eps_scale = max(0.01, req.epsilon)
    diff_visual = torch.clamp((diff_tensor + eps_scale) / (2.0 * eps_scale), 0.0, 1.0)

    # Compute distortion metrics
    metrics = compute_distortion_metrics(tensor, adv_tensor)

    return {
        "original_prediction": orig_pred,
        "original_confidence": orig_conf,
        "original_probabilities": [float(p) for p in orig_probs],
        "adversarial_prediction": adv_pred,
        "adversarial_confidence": adv_conf,
        "adversarial_probabilities": [float(p) for p in adv_probs],
        "attack_success": (adv_pred == req.target_label) if req.targeted else (adv_pred != label_val),
        "adversarial_image_base64": tensor_to_base64_png(adv_tensor),
        "perturbation_image_base64": tensor_to_base64_png(diff_visual),
        "metrics": metrics,
    }


@app.post("/api/defense")
def apply_defense(req: DefenseRequest):
    model = load_cached_model(req.model_name)
    tensor = base64_png_to_tensor(req.image_base64).to(DEVICE)

    if req.defense_type == "bit_depth":
        defense_layer = BitDepthReduction(step=int(req.param_value)).to(DEVICE)
    elif req.defense_type == "spatial_blur":
        defense_layer = SpatialSmoothing(sigma=float(req.param_value)).to(DEVICE)
    elif req.defense_type == "tv_denoise":
        defense_layer = TotalVariationDenoising(weight=float(req.param_value)).to(DEVICE)
    else:
        raise HTTPException(status_code=400, detail=f"Unknown defense {req.defense_type}")

    sanitized_tensor = defense_layer(tensor)

    with torch.no_grad():
        logits = model(sanitized_tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        pred = int(probs.argmax())
        conf = float(probs[pred])

    return {
        "sanitized_image_base64": tensor_to_base64_png(sanitized_tensor),
        "prediction": pred,
        "confidence": conf,
        "probabilities": [float(p) for p in probs],
    }


@app.post("/api/loss-landscape")
def get_loss_landscape(req: LandscapeRequest):
    model = load_cached_model(req.model_name)
    clean_tensor = base64_png_to_tensor(req.image_base64).to(DEVICE)
    adv_tensor = base64_png_to_tensor(req.adv_image_base64).to(DEVICE)
    lbl_tensor = torch.tensor([req.true_label], dtype=torch.long, device=DEVICE)

    data = compute_loss_landscape_1d(
        model=model,
        image=clean_tensor,
        label=lbl_tensor,
        adv_image=adv_tensor,
        num_points=req.num_points,
    )
    return data


# Mount static frontend
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    def serve_frontend():
        return FileResponse(os.path.join(STATIC_DIR, "index.html"))
