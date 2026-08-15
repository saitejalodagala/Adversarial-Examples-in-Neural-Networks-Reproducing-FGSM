from typing import Dict, Optional
import matplotlib.pyplot as plt
import numpy as np


def plot_loss_landscape_1d(
    landscape_data: Dict[str, any],
    title: str = "1D Loss Landscape along Adversarial Direction",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot 1D loss trajectory across gamma interpolation values.
    """
    gammas = landscape_data["gammas"]
    losses = landscape_data["losses"]
    true_label = landscape_data.get("true_label", "True")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(gammas, losses, color="#2563eb", linewidth=2.5, label="Cross-Entropy Loss")

    # Mark key points: Clean (gamma=0) and Adversarial (gamma=1)
    if 0.0 in gammas:
        idx0 = gammas.index(0.0)
        ax.scatter([0.0], [losses[idx0]], color="#10b981", s=80, zorder=5, label="Clean Input (gamma=0)")
    if 1.0 in gammas:
        idx1 = gammas.index(1.0)
        ax.scatter([1.0], [losses[idx1]], color="#ef4444", s=80, zorder=5, label="Adversarial Input (gamma=1)")

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Linear Interpolation Factor (gamma)", fontsize=11)
    ax.set_ylabel("Loss L(theta, x + gamma * delta, y)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper left")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_loss_landscape_2d(
    landscape_data: Dict[str, any],
    title: str = "2D Loss Landscape Geometry",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot 2D contour surface of loss landscape.
    """
    u_vals = landscape_data["u_vals"]
    v_vals = landscape_data["v_vals"]
    loss_grid = np.array(landscape_data["loss_grid"])

    U, V = np.meshgrid(u_vals, v_vals)

    fig, ax = plt.subplots(figsize=(7, 6))
    contour = ax.contourf(U, V, loss_grid.T, levels=30, cmap="viridis")
    fig.colorbar(contour, ax=ax, label="Loss")

    # Mark center
    ax.scatter([0], [0], color="white", edgecolors="black", s=100, label="Clean Sample")

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Adversarial Direction (d1)", fontsize=11)
    ax.set_ylabel("Orthogonal Direction (d2)", fontsize=11)
    ax.legend(loc="upper right")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig
