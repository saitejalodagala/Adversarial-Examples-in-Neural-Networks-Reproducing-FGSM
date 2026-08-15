import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """
    Standard Convolutional Neural Network for MNIST classification.
    Architecture exactly matches the baseline model in Goodfellow et al. reproduction experiments:
    - Conv2d(1, 32, kernel=3, pad=1) -> ReLU -> MaxPool2d(2)
    - Conv2d(32, 64, kernel=3, pad=1) -> ReLU -> MaxPool2d(2)
    - Linear(3136, 128) -> ReLU -> Linear(128, 10)
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 10):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),  # (B, 32, 28, 28)
            nn.ReLU(),
            nn.MaxPool2d(2),                                      # (B, 32, 14, 14)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),          # (B, 64, 14, 14)
            nn.ReLU(),
            nn.MaxPool2d(2),                                      # (B, 64, 7, 7)
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass computing unnormalized class logits."""
        features = self.conv(x)
        flattened = features.view(features.size(0), -1)
        logits = self.fc(flattened)
        return logits

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract penultimate feature embeddings before classification head."""
        features = self.conv(x)
        flattened = features.view(features.size(0), -1)
        return self.fc[0](flattened)
