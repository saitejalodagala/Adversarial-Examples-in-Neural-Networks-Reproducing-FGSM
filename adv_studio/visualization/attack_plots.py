from typing import List, Optional, Dict
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


def plot_adversarial_gallery(
    clean_images: torch.Tensor,
    adv_images: torch.Tensor,
    true_labels: torch.Tensor,
    clean_preds: torch.Tensor,
    adv_preds: torch.Tensor,
    clean_probs: Optional[torch.Tensor] = None,
    adv_probs: Optional[torch.Tensor] = None,
    num_samples: int = 5,
    save_path: Optional[str] = None,
    figsize: Optional[tuple] = None,
) -> plt.Figure:
    """
    Generate publication-ready comparison gallery:
    Column 1: Clean Input (Label, Confidence)
    Column 2: Normalized Perturbation Heatmap (delta = x_adv - x)
    Column 3: Adversarial Input (Flipped Label, Confidence)
    """
    n = min(num_samples, clean_images.size(0))
    fig, axes = plt.subplots(n, 3, figsize=figsize or (9, 3 * n))
    if n == 1:
        axes = np.expand_dims(axes, 0)

    for i in range(n):
        clean = clean_images[i].squeeze().cpu().numpy()
        adv = adv_images[i].squeeze().cpu().numpy()
        diff = adv - clean

        c_lbl = true_labels[i].item()
        c_pred = clean_preds[i].item()
        a_pred = adv_preds[i].item()

        c_conf = f" ({clean_probs[i, c_pred].item()*100:.1f}%)" if clean_probs is not None else ""
        a_conf = f" ({adv_probs[i, a_pred].item()*100:.1f}%)" if adv_probs is not None else ""

        # 1. Clean image
        axes[i, 0].imshow(clean, cmap="gray")
        axes[i, 0].set_title(f"Clean: {c_pred}{c_conf}\nTrue: {c_lbl}", fontsize=10)
        axes[i, 0].axis("off")

        # 2. Perturbation heatmap
        im_diff = axes[i, 1].imshow(diff, cmap="coolwarm", vmin=-0.3, vmax=0.3)
        axes[i, 1].set_title(f"Perturbation (diff)\nL_inf: {np.max(np.abs(diff)):.3f}", fontsize=10)
        axes[i, 1].axis("off")
        fig.colorbar(im_diff, ax=axes[i, 1], fraction=0.046, pad=0.04)

        # 3. Adversarial image
        color = "red" if a_pred != c_lbl else "green"
        axes[i, 2].imshow(adv, cmap="gray")
        axes[i, 2].set_title(f"Adversarial: {a_pred}{a_conf}", fontsize=10, color=color, fontweight="bold")
        axes[i, 2].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_robustness_curves(
    epsilons: List[float],
    results_by_model: Dict[str, List[float]],
    title: str = "Model Robustness vs Perturbation Magnitude",
    xlabel: str = "Epsilon (Perturbation Budget)",
    ylabel: str = "Accuracy (%)",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot Epsilon vs Robust Accuracy comparative curves.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    markers = ["o", "s", "^", "D", "v", "x", "*"]

    for i, (name, accuracies) in enumerate(results_by_model.items()):
        marker = markers[i % len(markers)]
        acc_pct = [a * 100 if a <= 1.0 else a for a in accuracies]
        ax.plot(epsilons, acc_pct, marker=marker, linewidth=2, label=name)

    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="best", frameon=True)
    ax.set_ylim(-2, 102)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig
