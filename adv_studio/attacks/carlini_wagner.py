from typing import Optional
import torch
import torch.nn as nn
import torch.optim as optim
from adv_studio.attacks.base import BaseAttack


class CarliniWagnerL2Attack(BaseAttack):
    """
    Carlini-Wagner L2 Optimization Attack from Carlini & Wagner (2017)
    "Towards Evaluating the Robustness of Neural Networks".

    Features:
    - Change of variables x = 0.5 * (tanh(w) + 1) for natural [0, 1] bounds
    - Margin loss with confidence parameter kappa
    - Adaptive Adam optimizer on perturbation parameters w
    """

    def __init__(
        self,
        model: nn.Module,
        c: float = 1.0,
        kappa: float = 0.0,
        steps: int = 50,
        lr: float = 0.01,
        device: Optional[torch.device] = None,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
        targeted: bool = False,
    ):
        super().__init__(model, device, clip_min, clip_max, targeted)
        self.c = float(c)
        self.kappa = float(kappa)
        self.steps = int(steps)
        self.lr = float(lr)

    def generate(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        target_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self.model.eval()
        imgs = images.clone().detach().to(self.device)
        lbls = labels.clone().detach().to(self.device)
        targets = target_labels.clone().detach().to(self.device) if target_labels is not None else None

        # Convert [0, 1] images to atanh space w: x = 0.5 * (tanh(w) + 1)
        # Numerical stability clamp:
        imgs_scaled = torch.clamp((imgs - self.clip_min) / (self.clip_max - self.clip_min), 1e-6, 1.0 - 1e-6)
        w = torch.atanh(2.0 * imgs_scaled - 1.0)
        w_param = nn.Parameter(w.clone().detach(), requires_grad=True)

        optimizer = optim.Adam([w_param], lr=self.lr)

        best_adv_imgs = imgs.clone().detach()
        best_l2 = torch.ones(imgs.size(0), device=self.device) * 1e9

        for _ in range(self.steps):
            adv_imgs = 0.5 * (torch.tanh(w_param) + 1.0) * (self.clip_max - self.clip_min) + self.clip_min
            logits = self.model(adv_imgs)

            # L2 distance squared
            l2_dist = torch.sum((adv_imgs - imgs) ** 2, dim=[1, 2, 3])

            # Margin loss
            if self.targeted:
                if targets is None:
                    raise ValueError("Target labels required for targeted CW attack.")
                target_logits = torch.gather(logits, 1, targets.unsqueeze(1)).squeeze(1)
                other_logits = logits.clone()
                other_logits.scatter_(1, targets.unsqueeze(1), -1e9)
                max_other, _ = torch.max(other_logits, dim=1)
                f_loss = torch.clamp(max_other - target_logits + self.kappa, min=0.0)
            else:
                true_logits = torch.gather(logits, 1, lbls.unsqueeze(1)).squeeze(1)
                other_logits = logits.clone()
                other_logits.scatter_(1, lbls.unsqueeze(1), -1e9)
                max_other, _ = torch.max(other_logits, dim=1)
                f_loss = torch.clamp(true_logits - max_other + self.kappa, min=0.0)

            total_loss = torch.mean(l2_dist + self.c * f_loss)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Track successful adversarial examples with minimal L2 distance
            with torch.no_grad():
                preds = torch.argmax(logits, dim=1)
                if self.targeted:
                    successful = (preds == targets)
                else:
                    successful = (preds != lbls)

                improved = successful & (l2_dist < best_l2)
                best_l2 = torch.where(improved, l2_dist, best_l2)
                improved_expanded = improved.view(-1, 1, 1, 1)
                best_adv_imgs = torch.where(improved_expanded, adv_imgs, best_adv_imgs)

        return self._clamp(best_adv_imgs)
