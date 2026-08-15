import torch
import torch.nn as nn
import torch.nn.functional as F


class BitDepthReduction(nn.Module):
    """
    Feature Squeezing: Bit-Depth Reduction (Xu et al., NDSS 2018).
    Quantizes pixel values from 8-bit (256 levels) to k bits (2^k levels)
    to eliminate subtle, low-amplitude adversarial perturbations.
    """

    def __init__(self, step: int = 4):
        super().__init__()
        # step represents number of bits (e.g. 1 to 8)
        self.step = max(1, min(8, int(step)))
        self.levels = 2 ** self.step - 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Quantize to 2^k levels
        scaled = torch.round(x * self.levels) / self.levels
        return torch.clamp(scaled, 0.0, 1.0)


class SpatialSmoothing(nn.Module):
    """
    Spatial Gaussian Blur Filter (Xu et al., 2018).
    Applies Gaussian convolution kernel to suppress high-frequency adversarial gradients.
    """

    def __init__(self, kernel_size: int = 3, sigma: float = 1.0, channels: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.channels = channels

        # Construct Gaussian kernel
        coords = torch.arange(kernel_size).float() - (kernel_size - 1) / 2.0
        grid_y, grid_x = torch.meshgrid(coords, coords, indexing="ij")
        kernel = torch.exp(-(grid_x ** 2 + grid_y ** 2) / (2.0 * sigma ** 2))
        kernel = kernel / torch.sum(kernel)

        kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)
        self.register_buffer("weight", kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        padding = self.kernel_size // 2
        return F.conv2d(x, self.weight.to(x.device), padding=padding, groups=self.channels)


class TotalVariationDenoising(nn.Module):
    """
    Total Variation (TV) Minimization / Denoising filter.
    Smooths adversarial perturbations while preserving sharp digit boundaries.
    """

    def __init__(self, weight: float = 0.05, iterations: int = 10):
        super().__init__()
        self.weight = weight
        self.iterations = iterations

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_denoised = x.clone().detach().requires_grad_(True)
        optimizer = torch.optim.SGD([x_denoised], lr=0.1)

        for _ in range(self.iterations):
            optimizer.zero_grad()
            fidelity = F.mse_loss(x_denoised, x)
            tv_h = torch.sum(torch.abs(x_denoised[:, :, 1:, :] - x_denoised[:, :, :-1, :]))
            tv_w = torch.sum(torch.abs(x_denoised[:, :, :, 1:] - x_denoised[:, :, :, :-1]))
            loss = fidelity + self.weight * (tv_h + tv_w) / x.numel()
            loss.backward()
            optimizer.step()

        return torch.clamp(x_denoised.detach(), 0.0, 1.0)
