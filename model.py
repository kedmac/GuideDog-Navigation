# model.py
import torch
import torch.nn as nn
from torchvision import models


class NavigationModel(nn.Module):
    def __init__(self, num_classes: int = 9, pretrained: bool = True):
        super().__init__()
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        backbone = models.mobilenet_v3_small(weights=weights)

        # Replace the final classifier layer
        in_features = backbone.classifier[-1].in_features
        backbone.classifier[-1] = nn.Linear(in_features, num_classes)

        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    @property
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def size_mb(self) -> float:
        return self.param_count * 4 / (1024 ** 2)
