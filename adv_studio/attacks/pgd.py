from typing import Optional
import torch
import torch.nn as nn
from adv_studio.attacks.base import BaseAttack


class PGDAttack(BaseAttack):
    """
    Projected Gradient Descent (PGD) from Madry et al. (2018)
    "Towards Deep Learning Models Resistant to Adversarial Attacks".

    Features:
    - L_inf and L_2 norm bounds
    - Random uniform start inside the epsilon-ball
    - Multiple random restarts
    - Targeted and untargeted optimization
    """

    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 0.2,
        alpha: Optional[float] = None,
        steps: int = 20,
        random_start: bool = True,
        norm: str = "Linf",
        restarts: int = 1,
        criterion: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
        targeted: bool = False,
    ):
        super().__init__(model, device, clip_min, clip_max, targeted)
        self.epsilon = float(epsilon)
        self.steps = int(steps)
        self.random_start = random_start
        self.norm = norm
        self.restarts = max(1, int(restarts))
        self.alpha = float(alpha) if alpha is not None else float(self.epsilon / 4.0 if norm == "Linf" else self.epsilon / 5.0)
        self.criterion = criterion or nn.CrossEntropyLoss()

    def generate(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        target_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self.model.eval()
        orig_imgs = images.clone().detach().to(self.device)
        lbls = labels.clone().detach().to(self.device)
        targets = target_labels.clone().detach().to(self.device) if target_labels is not None else None

        best_adv_imgs = orig_imgs.clone().detach()
        best_loss = -torch.ones(orig_imgs.size(0), device=self.device) * 1e9 if not self.targeted else torch.ones(orig_imgs.size(0), device=self.device) * 1e9

        for _ in range(self.restarts):
            adv_imgs = orig_imgs.clone().detach()

            if self.random_start:
                if self.norm == "Linf":
                    noise = torch.FloatTensor(*adv_imgs.shape).uniform_(-self.epsilon, self.epsilon).to(self.device)
                    adv_imgs = self._clamp(adv_imgs + noise)
                elif self.norm == "L2":
                    noise = torch.randn_like(adv_imgs).to(self.device)
                    noise_norm = torch.norm(noise.view(noise.size(0), -1), p=2, dim=1, keepdim=True).view(-1, 1, 1, 1) + 1e-8
                    r = torch.rand(adv_imgs.size(0), 1, 1, 1, device=self.device)
                    adv_imgs = self._clamp(adv_imgs + (noise / noise_norm) * (r * self.epsilon))

            for _ in range(self.steps):
                adv_imgs.requires_grad = True
                outputs = self.model(adv_imgs)

                if self.targeted:
                    if targets is None:
                        raise ValueError("Target labels must be provided for targeted PGD.")
                    loss_vec = nn.functional.cross_entropy(outputs, targets, reduction="none")
                    loss = loss_vec.sum()
                else:
                    loss_vec = nn.functional.cross_entropy(outputs, lbls, reduction="none")
                    loss = loss_vec.sum()

                self.model.zero_grad()
                loss.backward()

                if self.norm == "Linf":
                    grad = adv_imgs.grad.data.sign()
                    if self.targeted:
                        adv_imgs = adv_imgs.data - self.alpha * grad
                    else:
                        adv_imgs = adv_imgs.data + self.alpha * grad
                    eta = torch.clamp(adv_imgs - orig_imgs, -self.epsilon, self.epsilon)
                    adv_imgs = self._clamp(orig_imgs + eta).detach()

                elif self.norm == "L2":
                    grad = adv_imgs.grad.data
                    grad_norm = torch.norm(grad.view(grad.size(0), -1), p=2, dim=1, keepdim=True).view(-1, 1, 1, 1) + 1e-8
                    grad_unit = grad / grad_norm

                    if self.targeted:
                        adv_imgs = adv_imgs.data - self.alpha * grad_unit
                    else:
                        adv_imgs = adv_imgs.data + self.alpha * grad_unit

                    eta = adv_imgs - orig_imgs
                    eta_norm = torch.norm(eta.view(eta.size(0), -1), p=2, dim=1, keepdim=True).view(-1, 1, 1, 1) + 1e-8
                    factor = torch.min(torch.ones_like(eta_norm), self.epsilon / eta_norm)
                    adv_imgs = self._clamp(orig_imgs + eta * factor).detach()

            # Record best adversary across restarts
            with torch.no_grad():
                final_outputs = self.model(adv_imgs)
                if self.targeted:
                    curr_loss = nn.functional.cross_entropy(final_outputs, targets, reduction="none")
                    better_mask = curr_loss < best_loss
                else:
                    curr_loss = nn.functional.cross_entropy(final_outputs, lbls, reduction="none")
                    better_mask = curr_loss > best_loss

                best_loss = torch.where(better_mask, curr_loss, best_loss)
                better_mask_expanded = better_mask.view(-1, 1, 1, 1)
                best_adv_imgs = torch.where(better_mask_expanded, adv_imgs, best_adv_imgs)

        return best_adv_imgs
