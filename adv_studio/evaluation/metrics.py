from typing import Dict, Tuple, Optional
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from adv_studio.attacks.base import BaseAttack


def compute_distortion_metrics(clean_images: torch.Tensor, adv_images: torch.Tensor) -> Dict[str, float]:
    """
    Compute standard adversarial distortion metrics between clean and perturbed tensors.

    Metrics:
    - L_inf: max absolute difference
    - L_2: root mean squared / Euclidean norm per image
    - L_0: percentage of non-zero modified pixels (diff > 1e-4)
    - PSNR: Peak Signal-to-Noise Ratio (in dB)
    - SSIM: Structural Similarity Index Measure (approximate)
    """
    diff = (adv_images - clean_images).detach().cpu()
    batch_size = diff.size(0)

    # L_inf
    l_inf = torch.max(torch.abs(diff)).item()

    # L_2 per image
    l_2 = torch.mean(torch.norm(diff.view(batch_size, -1), p=2, dim=1)).item()

    # L_0 fraction
    l_0 = torch.mean((torch.abs(diff.view(batch_size, -1)) > 1e-4).float().sum(dim=1) / diff.view(batch_size, -1).size(1)).item()

    # MSE & PSNR
    mse = torch.mean(diff ** 2).item()
    if mse < 1e-10:
        psnr = 100.0
    else:
        psnr = 10.0 * math.log10(1.0 / mse)

    # Approximate SSIM
    c1 = (0.01) ** 2
    c2 = (0.03) ** 2
    mu_x = torch.mean(clean_images.cpu())
    mu_y = torch.mean(adv_images.cpu())
    sigma_x = torch.var(clean_images.cpu())
    sigma_y = torch.var(adv_images.cpu())
    sigma_xy = torch.mean((clean_images.cpu() - mu_x) * (adv_images.cpu() - mu_y))

    ssim = ((2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2))
    ssim_val = float(ssim.item())

    return {
        "l_inf": float(l_inf),
        "l_2": float(l_2),
        "l_0": float(l_0),
        "mse": float(mse),
        "psnr_db": float(psnr),
        "ssim": float(ssim_val),
    }


def compute_robust_accuracy(
    model: nn.Module,
    loader: DataLoader,
    attack: Optional[BaseAttack] = None,
    defense: Optional[nn.Module] = None,
    device: Optional[torch.device] = None,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    """
    Evaluate Clean and Robust Accuracy, Attack Success Rate, and Mean Confidence.
    """
    model.eval()
    dev = device or (next(model.parameters()).device if list(model.parameters()) else torch.device("cpu"))

    clean_correct = 0
    adv_correct = 0
    total = 0
    clean_conf_sum = 0.0
    adv_conf_sum = 0.0

    for i, (images, labels) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break

        images, labels = images.to(dev), labels.to(dev)

        # 1. Clean evaluation
        with torch.no_grad():
            clean_in = defense(images) if defense is not None else images
            clean_logits = model(clean_in)
            clean_probs = torch.softmax(clean_logits, dim=1)
            clean_preds = clean_logits.argmax(dim=1)
            clean_correct += (clean_preds == labels).sum().item()
            clean_conf_sum += torch.gather(clean_probs, 1, labels.unsqueeze(1)).sum().item()

        # 2. Adversarial evaluation
        if attack is not None:
            adv_images = attack.generate(images, labels)
            with torch.no_grad():
                adv_in = defense(adv_images) if defense is not None else adv_images
                adv_logits = model(adv_in)
                adv_probs = torch.softmax(adv_logits, dim=1)
                adv_preds = adv_logits.argmax(dim=1)
                adv_correct += (adv_preds == labels).sum().item()
                adv_conf_sum += torch.gather(adv_probs, 1, labels.unsqueeze(1)).sum().item()

        total += labels.size(0)

    clean_acc = clean_correct / total if total > 0 else 0.0
    robust_acc = (adv_correct / total) if attack is not None and total > 0 else clean_acc
    asr = (clean_correct - adv_correct) / clean_correct if attack is not None and clean_correct > 0 else 0.0
    mean_adv_conf = (adv_conf_sum / total) if attack is not None and total > 0 else (clean_conf_sum / total if total > 0 else 0.0)

    return {
        "clean_accuracy": float(clean_acc),
        "robust_accuracy": float(robust_acc),
        "attack_success_rate": float(asr),
        "mean_clean_confidence": float(clean_conf_sum / total) if total > 0 else 0.0,
        "mean_adv_confidence": float(mean_adv_conf),
        "total_samples": int(total),
    }
