from typing import Optional
import torch
import torch.nn as nn
from adv_studio.attacks.base import BaseAttack


class IFGSMAttack(BaseAttack):
    """
    Basic Iterative Method (I-FGSM / BIM) from Kurakin et al. (2016)
    "Adversarial examples in the physical world".

    Iterative update:
        x_0 = x
        x_{t+1} = Clip_{x, eps}( x_t + alpha * sign(grad_x Loss(theta, x_t, y)) )
    """

    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 0.2,
        alpha: Optional[float] = None,
        steps: int = 10,
        criterion: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
        targeted: bool = False,
    ):
        super().__init__(model, device, clip_min, clip_max, targeted)
        self.epsilon = float(epsilon)
        self.steps = int(steps)
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

        for _ in range(self.steps):
            adv_imgs.requires_grad = True
            outputs = self.model(adv_imgs)

            if self.targeted:
                if targets is None:
                    raise ValueError("Target labels must be provided for targeted I-FGSM.")
                loss = self.criterion(outputs, targets)
            else:
                loss = self.criterion(outputs, lbls)

            self.model.zero_grad()
            loss.backward()

            grad_sign = adv_imgs.grad.data.sign()

            if self.targeted:
                adv_imgs = adv_imgs.data - self.alpha * grad_sign
            else:
                adv_imgs = adv_imgs.data + self.alpha * grad_sign

            # Project perturbation back to L_inf epsilon ball
            eta = torch.clamp(adv_imgs - orig_imgs, -self.epsilon, self.epsilon)
            adv_imgs = self._clamp(orig_imgs + eta).detach()

        return adv_imgs
