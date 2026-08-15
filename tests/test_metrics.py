import pytest
import torch

from adv_studio.evaluation import compute_distortion_metrics


def test_distortion_metrics():
    clean = torch.zeros(2, 1, 28, 28)
    adv = torch.ones(2, 1, 28, 28) * 0.1

    metrics = compute_distortion_metrics(clean, adv)
    assert "l_inf" in metrics
    assert "l_2" in metrics
    assert "l_0" in metrics
    assert "psnr_db" in metrics
    assert "ssim" in metrics

    assert abs(metrics["l_inf"] - 0.1) < 1e-4
    assert metrics["l_0"] > 0.99
