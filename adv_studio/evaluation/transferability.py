from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from adv_studio.attacks.base import BaseAttack
from adv_studio.attacks.fgsm import FGSMAttack


def compute_transferability_matrix(
    models: Dict[str, nn.Module],
    loader: DataLoader,
    attack_fn_factory,
    epsilon: float = 0.2,
    device: Optional[torch.device] = None,
    max_batches: int = 5,
) -> Dict[str, any]:
    """
    Evaluate cross-model adversarial transferability.

    Args:
        models: Dictionary of named models { "SourceModel": model_instance, ... }
        loader: DataLoader with test images
        attack_fn_factory: Callable (model, epsilon, device) -> BaseAttack
        epsilon: Perturbation budget

    Returns:
        Dictionary containing model names, transfer matrix (ASR percentages), and robust accuracies.
    """
    model_names = list(models.keys())
    n = len(model_names)
    matrix = np.zeros((n, n), dtype=np.float32)

    dev = device or torch.device("cpu")

    for i, src_name in enumerate(model_names):
        src_model = models[src_name].to(dev).eval()
        attack = attack_fn_factory(src_model, epsilon=epsilon, device=dev)

        for images, labels in loader:
            images, labels = images.to(dev), labels.to(dev)
            adv_images = attack.generate(images, labels)

            for j, tgt_name in enumerate(model_names):
                tgt_model = models[tgt_name].to(dev).eval()
                with torch.no_grad():
                    clean_preds = tgt_model(images).argmax(dim=1)
                    adv_preds = tgt_model(adv_images).argmax(dim=1)

                    clean_correct = (clean_preds == labels).sum().item()
                    adv_flipped = ((clean_preds == labels) & (adv_preds != labels)).sum().item()

                    matrix[i, j] += adv_flipped

    # Normalize to fraction of successful clean classifications
    total_clean = sum(len(lbls) for _, lbls in loader)
    matrix = matrix / max(1, total_clean)

    return {
        "models": model_names,
        "transfer_matrix": matrix.tolist(),
        "epsilon": epsilon,
    }
