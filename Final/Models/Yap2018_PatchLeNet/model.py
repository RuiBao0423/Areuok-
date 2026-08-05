# -*- coding: utf-8 -*-
"""
Patch-based LeNet (Yap et al., IEEE JBHI 2018, "Automated Breast Ultrasound Lesions
Detection Using CNNs", Fig. 3) --- our from-scratch PyTorch implementation.

The paper classifies 28x28 grayscale patches as lesion / non-lesion:
  conv 5x5 (20 maps) -> max-pool 2x2 -> conv 5x5 (50 maps) -> max-pool 2x2
  -> FC 500 (ReLU, dropout) -> FC 2 (softmax).
At test time a sliding window over the image produces a lesion probability map.
"""
import torch
import torch.nn as nn


class PatchLeNet(nn.Module):
    """LeNet as in Yap et al. Fig. 3. We add BatchNorm (not in the 2018 paper) to
    stabilise training and lift patch accuracy, which cuts test-time false positives."""
    def __init__(self, dropout=0.33):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 20, 5), nn.BatchNorm2d(20), nn.ReLU(inplace=True), nn.MaxPool2d(2),   # 28->24->12
            nn.Conv2d(20, 50, 5), nn.BatchNorm2d(50), nn.ReLU(inplace=True), nn.MaxPool2d(2),  # 12->8->4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(50*4*4, 500), nn.BatchNorm1d(500), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(500, 2),
        )

    def forward(self, x):
        return self.classifier(self.features(x))          # logits (B,2)


if __name__ == "__main__":
    m = PatchLeNet()
    x = torch.randn(4, 1, 28, 28)
    print("output:", m(x).shape, "| params:", sum(p.numel() for p in m.parameters())/1e6, "M")
