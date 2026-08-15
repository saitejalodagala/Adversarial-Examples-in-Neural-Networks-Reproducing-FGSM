import pytest
import torch

from adv_studio.defenses import (
    BitDepthReduction,
    SpatialSmoothing,
    TotalVariationDenoising,
    RandomizedSmoothing,
)
from adv_studio.models import SimpleCNN


def test_bit_depth_reduction():
    layer = BitDepthReduction(step=2)  # 4 levels: 0, 1/3, 2/3, 1
    x = torch.tensor([[[[0.1, 0.4], [0.7, 0.95]]]])
    out = layer(x)
    assert out.shape == x.shape
    assert torch.all(out >= 0.0)
    assert torch.all(out <= 1.0)


def test_spatial_smoothing():
    layer = SpatialSmoothing(kernel_size=3, sigma=1.0)
    x = torch.rand(2, 1, 28, 28)
    out = layer(x)
    assert out.shape == x.shape


def test_tv_denoising():
    layer = TotalVariationDenoising(weight=0.05, iterations=3)
    x = torch.rand(2, 1, 28, 28)
    out = layer(x)
    assert out.shape == x.shape
    assert torch.all(out >= 0.0)
    assert torch.all(out <= 1.0)


def test_randomized_smoothing():
    model = SimpleCNN()
    smoothing = RandomizedSmoothing(model, sigma=0.25)
    x = torch.rand(2, 1, 28, 28)
    preds, counts = smoothing.predict(x, num_samples=10)
    assert preds.shape == (2,)
    assert counts.shape == (2, 10)
