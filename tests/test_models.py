import pytest
import torch

from adv_studio.models import SimpleCNN, LeNet5, MiniResNet18, MLP, get_model


def test_simple_cnn_forward():
    model = SimpleCNN()
    x = torch.rand(2, 1, 28, 28)
    out = model(x)
    assert out.shape == (2, 10)


def test_lenet_forward():
    model = LeNet5()
    x = torch.rand(2, 1, 28, 28)
    out = model(x)
    assert out.shape == (2, 10)


def test_resnet_forward():
    model = MiniResNet18(in_channels=1, num_classes=10)
    x = torch.rand(2, 1, 28, 28)
    out = model(x)
    assert out.shape == (2, 10)


def test_mlp_forward():
    model = MLP()
    x = torch.rand(2, 1, 28, 28)
    out = model(x)
    assert out.shape == (2, 10)


def test_model_factory():
    m1 = get_model("simple_cnn")
    assert isinstance(m1, SimpleCNN)
    m2 = get_model("lenet")
    assert isinstance(m2, LeNet5)
    m3 = get_model("resnet")
    assert isinstance(m3, MiniResNet18)
    m4 = get_model("mlp")
    assert isinstance(m4, MLP)
