from typing import Optional, Tuple
import math
import numpy as np
from scipy.stats import norm
import torch
import torch.nn as nn


class RandomizedSmoothing:
    """
    Certified Adversarial Robustness via Randomized Smoothing (Cohen et al., 2019).

    Transforms base classifier f into smoothed classifier g:
        g(x) = argmax_c P( f(x + epsilon) = c ), where epsilon ~ N(0, sigma^2 * I)

    Certified L2 Radius:
        R = (sigma / 2) * ( Phi^{-1}(p_A) - Phi^{-1}(p_B) )
    """

    def __init__(
        self,
        base_model: nn.Module,
        sigma: float = 0.25,
        num_classes: int = 10,
        device: Optional[torch.device] = None,
    ):
        self.base_model = base_model
        self.sigma = float(sigma)
        self.num_classes = num_classes
        self.device = device or (next(base_model.parameters()).device if list(base_model.parameters()) else torch.device("cpu"))

    def predict(self, images: torch.Tensor, num_samples: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate the most probable class for each image by sampling Gaussian noise.

        Returns:
            (predicted_classes, class_counts)
        """
        self.base_model.eval()
        imgs = images.to(self.device)
        batch_size = imgs.size(0)

        counts = torch.zeros((batch_size, self.num_classes), device=self.device)

        with torch.no_grad():
            for _ in range(num_samples):
                noise = torch.randn_like(imgs) * self.sigma
                noisy_imgs = imgs + noise
                logits = self.base_model(noisy_imgs)
                preds = torch.argmax(logits, dim=1)
                counts.scatter_add_(1, preds.unsqueeze(1), torch.ones_like(preds.unsqueeze(1), dtype=torch.float))

        predicted_classes = torch.argmax(counts, dim=1)
        return predicted_classes, counts

    def certify(self, image: torch.Tensor, n0: int = 100, n: int = 1000, alpha: float = 0.001) -> Tuple[int, float]:
        """
        Certify the prediction for a single image with confidence 1 - alpha.

        Returns:
            (predicted_class, certified_radius_L2)
        """
        self.base_model.eval()
        img = image.to(self.device)

        # 1. Selection step with n0 samples
        c_hat, _ = self.predict(img, num_samples=n0)
        c_hat_val = c_hat.item()

        # 2. Estimation step with n samples
        _, counts = self.predict(img, num_samples=n)
        count_c_hat = counts[0, c_hat_val].item()

        # Lower bound on probability of top class
        # Clopper-Pearson confidence interval
        from scipy.stats import beta
        p_A_lower = beta.ppf(alpha, count_c_hat, n - count_c_hat + 1)

        if p_A_lower > 0.5:
            radius = self.sigma * norm.ppf(p_A_lower)
            return c_hat_val, float(radius)
        else:
            return -1, 0.0
