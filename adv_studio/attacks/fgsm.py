from typing import Optional
import torch
import torch.nn as nn
from adv_studio.attacks.base import BaseAttack


class FGSMAttack(BaseAttack):
    """
    Fast Gradient Sign Method (FGSM) from Goodfellow et al. (2014)
    "Explaining and Harnessing Adversarial Examples".

    Untargeted Formulation:
        x_adv = x + epsilon * sign(grad_x Loss(theta, x, y))

    Targeted Formulation:
        x_adv = x - epsilon * sign(grad_x Loss(theta, x, y_target))
    """

    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 0.2,
        criterion: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
        targeted: bool = False,
    ):
        super().__init__(model, device, clip_min, clip_max, targeted)
        self.epsilon = float(epsilon)
        self.criterion = criterion or nn.CrossEntropyLoss()

    def generate(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        target_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self.model.eval()
        imgs = images.clone().detach().to(self.device)
        lbls = labels.clone().detach().to(self.device)
        imgs.requires_grad = True

        outputs = self.model(imgs)

        if self.targeted:
            if target_labels is None:
                raise ValueError("Target labels must be provided for targeted FGSM attack.")
            targets = target_labels.clone().detach().to(self.device)
            loss = self.criterion(outputs, targets)
        else:
            loss = self.criterion(outputs, lbls)

        self.model.zero_grad()
        loss.backward()

        if imgs.grad is None:
            raise RuntimeError("Gradient computation failed in FGSM attack.")

        grad_sign = imgs.grad.data.sign()

        if self.targeted:
            adv_images = imgs.data - self.epsilon * grad_sign
        else:
            adv_images = imgs.data + self.epsilon * grad_sign

        return self._clamp(adv_images)


class TargetedFGSMAttack(FGSMAttack):
    """Convenience subclass for Targeted FGSM attacks."""

    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 0.2,
        criterion: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
    ):
        super().__init__(
            model=model,
            epsilon=epsilon,
            criterion=criterion,
            device=device,
            clip_min=clip_min,
            clip_max=clip_max,
            targeted=True,
        )
