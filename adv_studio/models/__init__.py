import os
import torch
import torch.nn as nn
from typing import Optional

from adv_studio.models.simple_cnn import SimpleCNN
from adv_studio.models.lenet import LeNet5
from adv_studio.models.resnet import MiniResNet18
from adv_studio.models.mlp import MLP

MODEL_REGISTRY = {
    "simple_cnn": SimpleCNN,
    "cnn": SimpleCNN,
    "lenet": LeNet5,
    "lenet5": LeNet5,
    "resnet": MiniResNet18,
    "miniresnet": MiniResNet18,
    "mlp": MLP,
}


def get_model(
    name: str = "simple_cnn",
    in_channels: int = 1,
    num_classes: int = 10,
    pretrained_path: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """
    Model factory to instantiate and optionally load pretrained weights.
    """
    key = name.lower().strip()
    if key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model name '{name}'. Available: {list(MODEL_REGISTRY.keys())}")

    model_class = MODEL_REGISTRY[key]
    if key == "mlp":
        model = model_class(in_features=in_channels * 28 * 28, num_classes=num_classes)
    else:
        model = model_class(in_channels=in_channels, num_classes=num_classes)

    if pretrained_path and os.path.exists(pretrained_path):
        state_dict = torch.load(pretrained_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)

    if device is not None:
        model = model.to(device)

    return model
