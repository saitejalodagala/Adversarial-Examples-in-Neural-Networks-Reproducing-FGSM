import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    3-Layer Multi-Layer Perceptron (MLP) baseline for testing the Linearity Hypothesis:
    - Linear(784, 512) -> ReLU -> Dropout(0.2)
    - Linear(512, 256) -> ReLU -> Dropout(0.2)
    - Linear(256, 10)
    """

    def __init__(self, in_features: int = 784, num_classes: int = 10, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
