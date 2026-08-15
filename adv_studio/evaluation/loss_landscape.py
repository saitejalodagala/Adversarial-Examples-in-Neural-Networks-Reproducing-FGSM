from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn


def compute_loss_landscape_1d(
    model: nn.Module,
    image: torch.Tensor,
    label: torch.Tensor,
    adv_image: torch.Tensor,
    gamma_min: float = -1.0,
    gamma_max: float = 2.0,
    num_points: int = 50,
) -> Dict[str, any]:
    """
    Compute 1D Loss Landscape along the 1D line passing through clean image x and adversarial image x_adv.
    x(gamma) = x + gamma * (x_adv - x)

    Demonstrates Goodfellow's Linearity Hypothesis: If the loss is linear along the perturbation vector,
    the model behaves linearly in high-dimensional space.
    """
    model.eval()
    dev = next(model.parameters()).device if list(model.parameters()) else torch.device("cpu")

    img = image.clone().detach().to(dev)
    adv = adv_image.clone().detach().to(dev)
    lbl = label.clone().detach().to(dev)

    delta = adv - img
    gammas = np.linspace(gamma_min, gamma_max, num_points).tolist()

    losses = []
    probabilities = []
    predictions = []

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for gamma in gammas:
            interp = torch.clamp(img + gamma * delta, 0.0, 1.0)
            logits = model(interp)
            loss = criterion(logits, lbl).item()
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy().tolist()
            pred = logits.argmax(dim=1).item()

            losses.append(float(loss))
            probabilities.append(probs)
            predictions.append(int(pred))

    return {
        "gammas": gammas,
        "losses": losses,
        "probabilities": probabilities,
        "predictions": predictions,
        "true_label": int(lbl.item() if lbl.dim() == 1 else lbl[0].item()),
    }


def compute_loss_landscape_2d(
    model: nn.Module,
    image: torch.Tensor,
    label: torch.Tensor,
    adv_image: torch.Tensor,
    grid_size: int = 21,
    range_lim: float = 1.5,
) -> Dict[str, any]:
    """
    Compute 2D Loss Landscape grid along adversarial gradient direction (d1)
    and an orthogonal random direction (d2).
    """
    model.eval()
    dev = next(model.parameters()).device if list(model.parameters()) else torch.device("cpu")

    img = image.clone().detach().to(dev)
    adv = adv_image.clone().detach().to(dev)
    lbl = label.clone().detach().to(dev)

    # Direction 1: Normalized adversarial vector
    d1 = adv - img
    d1_norm = torch.norm(d1.view(-1), p=2) + 1e-8
    d1 = d1 / d1_norm

    # Direction 2: Orthogonalized random Gaussian vector
    rand_vec = torch.randn_like(img).to(dev)
    proj = torch.sum(rand_vec * d1) * d1
    d2 = rand_vec - proj
    d2_norm = torch.norm(d2.view(-1), p=2) + 1e-8
    d2 = d2 / d2_norm

    u_vals = np.linspace(-range_lim, range_lim, grid_size).tolist()
    v_vals = np.linspace(-range_lim, range_lim, grid_size).tolist()

    grid_loss = np.zeros((grid_size, grid_size), dtype=np.float32)
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for i, u in enumerate(u_vals):
            for j, v in enumerate(v_vals):
                perturbed = torch.clamp(img + u * d1 + v * d2, 0.0, 1.0)
                logits = model(perturbed)
                loss = criterion(logits, lbl).item()
                grid_loss[i, j] = float(loss)

    return {
        "u_vals": u_vals,
        "v_vals": v_vals,
        "loss_grid": grid_loss.tolist(),
        "true_label": int(lbl.item() if lbl.dim() == 1 else lbl[0].item()),
    }
