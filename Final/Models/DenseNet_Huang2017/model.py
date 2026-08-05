"""DenseNet-121 classifier for the BUSI three-class classification task."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import DenseNet121_Weights, densenet121


class DenseNet121Classifier(nn.Module):
    """ImageNet DenseNet-121 with a BUSI classification head.

    Expected input shape is ``(N, 3, 256, 256)`` and the output contains
    unnormalised logits for ``normal``, ``benign`` and ``malignant``.
    """

    def __init__(
        self,
        num_classes: int = 3,
        pretrained: bool = True,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        weights = DenseNet121_Weights.DEFAULT if pretrained else None
        self.backbone = densenet121(weights=weights)
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def build_model(
    num_classes: int = 3,
    pretrained: bool = True,
    dropout: float = 0.2,
) -> DenseNet121Classifier:
    """Factory used by the training script and external evaluation code."""
    return DenseNet121Classifier(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
    )


if __name__ == "__main__":
    model = build_model(pretrained=False)
    with torch.inference_mode():
        output = model(torch.randn(2, 3, 256, 256))
    print(model)
    print("output shape:", tuple(output.shape))
