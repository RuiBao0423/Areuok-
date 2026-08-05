# -*- coding: utf-8 -*-
"""
ResNet (He, Zhang, Ren & Sun, "Deep Residual Learning for Image Recognition", CVPR 2016)
-- our transfer-learning reproduction for 3-class BUSI classification
(normal / benign / malignant).

Following exactly the transfer-learning recipe the project brief points at (and mirroring
our FCN-AlexNet baseline): take a torchvision ResNet pretrained on ImageNet -- the residual
architecture of the paper (Fig. 3 right, Table 1: conv1 -> 4 stages of residual blocks ->
global average pool -> fc) -- and replace its 1000-way classifier head with a `n_class`-way
linear layer, then fine-tune on our leakage-free BUSI split.

We default to ResNet-18 (the smallest variant in Table 1) because BUSI is tiny (~780 images);
`arch="resnet50"` selects the bottleneck 50-layer variant if more capacity is wanted. BUSI is
grayscale, replicated to 3 channels (pseudo-RGB) so the pretrained conv1 weights transfer.
"""
import torch
import torch.nn as nn
import torchvision

_ARCH = {
    "resnet18": (torchvision.models.resnet18, torchvision.models.ResNet18_Weights.IMAGENET1K_V1),
    "resnet50": (torchvision.models.resnet50, torchvision.models.ResNet50_Weights.IMAGENET1K_V1),
}


class ResNetClassifier(nn.Module):
    def __init__(self, n_class=3, arch="resnet18", pretrained=True):
        super().__init__()
        if arch not in _ARCH:
            raise ValueError(f"arch must be one of {list(_ARCH)}, got {arch!r}")
        ctor, weights = _ARCH[arch]
        self.arch = arch
        self.net = ctor(weights=weights if pretrained else None)
        in_f = self.net.fc.in_features
        self.net.fc = nn.Linear(in_f, n_class)        # swap 1000-way head -> n_class head

    def forward(self, x):
        return self.net(x)                            # logits (B, n_class)


if __name__ == "__main__":
    m = ResNetClassifier(3, arch="resnet18", pretrained=False)
    x = torch.randn(2, 3, 256, 256)
    print("output:", m(x).shape, "| params:", sum(p.numel() for p in m.parameters())/1e6, "M")
