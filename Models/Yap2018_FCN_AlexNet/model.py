# -*- coding: utf-8 -*-
"""
FCN-AlexNet (transfer learning) -- Yap et al., IEEE JBHI 2018, Section IV-A-3, the
paper's BEST method. Our from-scratch PyTorch reproduction of the idea: take an
ImageNet-pretrained AlexNet, make it fully convolutional, and fine-tune it for
lesion segmentation (Long et al. 2015 "Fully Convolutional Networks", the FCN the
paper builds on).

- Backbone: torchvision AlexNet convolutional features, pretrained on ImageNet
  (this is the "transfer learning" the paper relies on to overcome data scarcity).
- Head: fc6/fc7 re-cast as 1x1/3x3 conv layers (fully convolutional) + a 1x1 score
  conv, then bilinear upsampling back to the input resolution (FCN-32s style).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class FCNAlexNet(nn.Module):
    def __init__(self, n_class=1, dropout=0.33, pretrained=True):
        super().__init__()
        weights = torchvision.models.AlexNet_Weights.IMAGENET1K_V1 if pretrained else None
        alex = torchvision.models.alexnet(weights=weights)
        self.backbone = alex.features                 # pretrained conv1-5 (stride 32)
        self.head = nn.Sequential(                     # fc6, fc7, score -> fully convolutional
            nn.Conv2d(256, 4096, 3, padding=1), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Conv2d(4096, 4096, 1),           nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Conv2d(4096, n_class, 1),
        )

    def forward(self, x):
        h, w = x.shape[-2:]
        f = self.backbone(x)                           # (B,256,H/32,W/32)
        s = self.head(f)                               # (B,1,H/32,W/32)
        return F.interpolate(s, size=(h, w), mode="bilinear", align_corners=False)


if __name__ == "__main__":
    m = FCNAlexNet()
    x = torch.randn(2, 3, 256, 256)
    print("output:", m(x).shape, "| params:", sum(p.numel() for p in m.parameters())/1e6, "M")
