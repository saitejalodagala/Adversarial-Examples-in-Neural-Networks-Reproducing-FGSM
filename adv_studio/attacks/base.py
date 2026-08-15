from abc import ABC, abstractmethod
from typing import Optional, Tuple
import torch
import torch.nn as nn


class BaseAttack(ABC):
    """
    Abstract base class for all adversarial attacks.
    Ensures consistent API across untargeted, targeted, white-box, and black-box attacks.
    """

    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
        targeted: bool = False,
    ):
        self.model = model
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.targeted = targeted
        if device is None:
            self.device = next(model.parameters()).device if list(model.parameters()) else torch.device("cpu")
        else:
            self.device = device

    @abstractmethod
    def generate(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        target_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Generate adversarial examples.

        Args:
            images: Batch of clean images (B, C, H, W) normalized to [clip_min, clip_max].
            labels: Ground truth class labels (B,).
            target_labels: Desired target class labels (B,) if targeted=True.

        Returns:
            Adversarial images (B, C, H, W) clamped to [clip_min, clip_max].
        """
        pass

    def __call__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        target_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.generate(images, labels, target_labels)

    def _clamp(self, adv_images: torch.Tensor) -> torch.Tensor:
        return torch.clamp(adv_images, self.clip_min, self.clip_max)
