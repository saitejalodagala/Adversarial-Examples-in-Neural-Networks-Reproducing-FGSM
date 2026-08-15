from typing import Optional
import torch
import torch.nn as nn
from adv_studio.attacks.base import BaseAttack


class RandomNoiseAttack(BaseAttack):
    """
    Random Noise baseline perturbation (Uniform or Gaussian).
    Used to empirically verify that neural network vulnerability is caused by
    directed adversarial gradient alignments rather than arbitrary random noise.
    """

    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 0.2,
        noise_type: str = "uniform",  # 'uniform' or 'gaussian'
        device: Optional[torch.device] = None,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
    ):
        super().__init__(model, device, clip_min, clip_max, targeted=False)
        self.epsilon = float(epsilon)
        self.noise_type = noise_type.lower()

    def generate(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        target_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        imgs = images.clone().detach().to(self.device)

        if self.noise_type == "gaussian":
            noise = torch.randn_like(imgs) * (self.epsilon / 2.0)
            noise = torch.clamp(noise, -self.epsilon, self.epsilon)
        else:
            noise = torch.FloatTensor(*imgs.shape).uniform_(-self.epsilon, self.epsilon).to(self.device)

        adv_imgs = self._clamp(imgs + noise)
        return adv_imgs
