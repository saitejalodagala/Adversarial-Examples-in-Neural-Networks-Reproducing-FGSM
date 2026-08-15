from typing import Optional
import torch
import torch.nn as nn
from adv_studio.attacks.base import BaseAttack


class MomentumFGSMAttack(BaseAttack):
    """
    Momentum Iterative Fast Gradient Sign Method (MI-FGSM) from Dong et al. (2018)
    "Boosting Adversarial Attacks with Momentum".

    Accumulates a velocity vector along the gradient path to stabilize updates
    and escape local extrema, significantly boosting black-box transferability.
    """

    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 0.2,
        alpha: Optional[float] = None,
        steps: int = 10,
        decay_factor: float = 1.0,
        criterion: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
        targeted: bool = False,
    ):
        super().__init__(model, device, clip_min, clip_max, targeted)
        self.epsilon = float(epsilon)
        self.steps = int(steps)
        self.decay_factor = float(decay_factor)
        self.alpha = float(alpha) if alpha is not None else float(self.epsilon / max(1, self.steps))
        self.criterion = criterion or nn.CrossEntropyLoss()

    def generate(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        target_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self.model.eval()
        orig_imgs = images.clone().detach().to(self.device)
        adv_imgs = orig_imgs.clone().detach()
        lbls = labels.clone().detach().to(self.device)
        targets = target_labels.clone().detach().to(self.device) if target_labels is not None else None

        momentum = torch.zeros_like(orig_imgs, device=self.device)

        for _ in range(self.steps):
            adv_imgs.requires_grad = True
            outputs = self.model(adv_imgs)

            if self.targeted:
                if targets is None:
                    raise ValueError("Target labels must be provided for targeted MI-FGSM.")
                loss = self.criterion(outputs, targets)
            else:
                loss = self.criterion(outputs, lbls)

            self.model.zero_grad()
            loss.backward()

            grad = adv_imgs.grad.data
            grad_l1 = torch.mean(torch.abs(grad), dim=[1, 2, 3], keepdim=True) + 1e-8
            grad_normalized = grad / grad_l1

            momentum = self.decay_factor * momentum + grad_normalized

            if self.targeted:
                adv_imgs = adv_imgs.data - self.alpha * torch.sign(momentum)
            else:
                adv_imgs = adv_imgs.data + self.alpha * torch.sign(momentum)

            eta = torch.clamp(adv_imgs - orig_imgs, -self.epsilon, self.epsilon)
            adv_imgs = self._clamp(orig_imgs + eta).detach()

        return adv_imgs
