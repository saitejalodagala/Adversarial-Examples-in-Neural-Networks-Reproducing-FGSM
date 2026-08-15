import torch
import torch.nn as nn


class LeNet5(nn.Module):
    """
    Classic LeNet-5 architecture adapted for modern MNIST classification:
    - Conv2d(1, 6, kernel=5) -> ReLU -> MaxPool2d(2)
    - Conv2d(6, 16, kernel=5) -> ReLU -> MaxPool2d(2)
    - Linear(16*4*4=256, 120) -> ReLU -> Linear(120, 84) -> ReLU -> Linear(84, 10)
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 10):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=5, padding=2),  # 28x28
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                # 14x14
            nn.Conv2d(6, 16, kernel_size=5),                      # 10x10
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                # 5x5
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        feat = feat.view(feat.size(0), -1)
        return self.classifier(feat)
