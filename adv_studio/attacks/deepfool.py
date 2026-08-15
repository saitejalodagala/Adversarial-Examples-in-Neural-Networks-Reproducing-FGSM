from typing import Optional
import torch
import torch.nn as nn
from adv_studio.attacks.base import BaseAttack


class DeepFoolAttack(BaseAttack):
    """
    DeepFool: A Simple and Accurate Method to Fool Deep Neural Networks
    (Moosavi-Dezfooli et al., 2016).

    Finds the minimal adversarial perturbation by iteratively linearizing
    the classifier boundaries and projecting towards the nearest hyperplane.
    """

    def __init__(
        self,
        model: nn.Module,
        max_iter: int = 50,
        overshoot: float = 0.02,
        device: Optional[torch.device] = None,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
    ):
        super().__init__(model, device, clip_min, clip_max, targeted=False)
        self.max_iter = int(max_iter)
        self.overshoot = float(overshoot)

    def generate(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        target_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self.model.eval()
        adv_batch = []

        for i in range(images.size(0)):
            x = images[i : i + 1].clone().detach().to(self.device)
            orig_lbl = labels[i].item()

            x_adv = x.clone().detach()

            for _ in range(self.max_iter):
                x_adv.requires_grad = True
                logits = self.model(x_adv)
                pred_lbl = torch.argmax(logits, dim=1).item()

                if pred_lbl != orig_lbl:
                    break

                num_classes = logits.size(1)
                orig_logit = logits[0, orig_lbl]
                orig_logit.backward(retain_graph=True)
                grad_orig = x_adv.grad.data.clone()

                min_dist = float("inf")
                best_w = None

                for k in range(num_classes):
                    if k == orig_lbl:
                        continue

                    self.model.zero_grad()
                    x_adv.grad.data.zero_()
                    other_logit = logits[0, k]
                    other_logit.backward(retain_graph=True)
                    grad_k = x_adv.grad.data.clone()

                    w_k = grad_k - grad_orig
                    f_k = (logits[0, k] - logits[0, orig_lbl]).data.item()

                    w_k_norm = torch.norm(w_k.view(-1), p=2).item() + 1e-8
                    dist_k = abs(f_k) / w_k_norm

                    if dist_k < min_dist:
                        min_dist = dist_k
                        best_w = (abs(f_k) / (w_k_norm ** 2)) * w_k

                if best_w is not None:
                    perturbation = (1.0 + self.overshoot) * best_w
                    x_adv = self._clamp(x_adv.data + perturbation).detach()

            adv_batch.append(x_adv.detach())

        return torch.cat(adv_batch, dim=0)
