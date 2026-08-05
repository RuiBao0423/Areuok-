# -*- coding: utf-8 -*-
"""
9444_07_deeplab/model.py
DeepLabV3+ (Chen et al. 2018, arXiv:1802.02611 -- "Encoder-Decoder with Atrous
Separable Convolution for Semantic Image Segmentation") for BUSI binary lesion
segmentation. Only the NETWORK is defined here; training lives in train.py.

Architecture (paper, adapted to a ResNet-50 backbone instead of Xception):
  Encoder:
    * ResNet-50 backbone (ImageNet-pretrained). The last stage's stride is
      replaced by atrous convolution so the output stride is 16 (feature map is
      1/16 of the input instead of 1/32) -- this keeps spatial detail while
      preserving the pretrained receptive field.
    * ASPP: 1x1 conv + three 3x3 atrous *separable* convs at rates 6/12/18
      (the standard rates for output stride 16) + an image-level global-pooling
      branch; all five concatenated and projected back to 256 channels.
  Decoder (the "+" of v3+):
    * take the low-level feature from layer1 (output stride 4), 1x1-conv it down
      to 48 channels, bilinearly upsample the ASPP output x4, concat, refine with
      two 3x3 separable convs, then upsample x4 back to the input resolution.
  Depthwise-separable convolution is used in ASPP and the decoder (the "Atrous
  Separable Convolution" of the title) to cut parameters.

I/O (matches common/data.py SegDataset with rgb3=True):
  input : (B, 3, 256, 256)  ImageNet-normalised grayscale->pseudo-RGB
  output: (B, 1, 256, 256)  raw logits  -> use with BCEWithLogitsLoss
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


class SeparableConv2d(nn.Module):
    """Depthwise-separable conv: a depthwise (optionally atrous) 3x3 followed by a
    pointwise 1x1, each with BN+ReLU. This is the 'separable' part of the paper's
    atrous separable convolution -- far fewer params than a dense 3x3."""
    def __init__(self, cin, cout, kernel=3, dilation=1):
        super().__init__()
        pad = dilation * (kernel - 1) // 2
        self.depthwise = nn.Conv2d(cin, cin, kernel, padding=pad,
                                   dilation=dilation, groups=cin, bias=False)
        self.bn1 = nn.BatchNorm2d(cin)
        self.pointwise = nn.Conv2d(cin, cout, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.depthwise(x)))
        x = self.relu(self.bn2(self.pointwise(x)))
        return x


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling. Five parallel branches probe the feature
    map at multiple fields-of-view, then get fused:
      b0 : 1x1 conv
      b1 : 3x3 atrous separable conv, rate 6
      b2 : 3x3 atrous separable conv, rate 12
      b3 : 3x3 atrous separable conv, rate 18
      pool: image-level global average pooling -> 1x1 conv -> upsample back
    Concatenated (5*out_ch) and projected to out_ch with a 1x1 conv + dropout."""
    def __init__(self, in_ch, out_ch=256, rates=(6, 12, 18)):
        super().__init__()
        self.b0 = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False),
                                nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.b1 = SeparableConv2d(in_ch, out_ch, 3, dilation=rates[0])
        self.b2 = SeparableConv2d(in_ch, out_ch, 3, dilation=rates[1])
        self.b3 = SeparableConv2d(in_ch, out_ch, 3, dilation=rates[2])
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                  nn.Conv2d(in_ch, out_ch, 1, bias=False),
                                  nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.project = nn.Sequential(
            nn.Conv2d(5 * out_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True), nn.Dropout(0.5))

    def forward(self, x):
        size = x.shape[-2:]
        p = self.pool(x)
        p = F.interpolate(p, size=size, mode="bilinear", align_corners=False)
        x = torch.cat([self.b0(x), self.b1(x), self.b2(x), self.b3(x), p], dim=1)
        return self.project(x)


class DeepLabV3Plus(nn.Module):
    """DeepLabV3+ with a dilated ResNet-50 encoder + ASPP + a light decoder."""
    def __init__(self, num_classes=1, output_stride=16, pretrained=True):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        # output stride 16 -> dilate only the last stage (rates 6/12/18 in ASPP);
        # output stride 8  -> dilate the last two stages (would want rates 12/24/36).
        if output_stride == 16:
            replace = [False, False, True]
            aspp_rates = (6, 12, 18)
        elif output_stride == 8:
            replace = [False, True, True]
            aspp_rates = (12, 24, 36)
        else:
            raise ValueError("output_stride must be 8 or 16")
        backbone = resnet50(weights=weights, replace_stride_with_dilation=replace)

        # keep stages separate so we can tap the low-level (output stride 4) feature
        self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1,
                                    backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1     # output stride 4,  256 channels (decoder skip)
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4     # output stride 16, 2048 channels (-> ASPP)

        self.aspp = ASPP(2048, 256, rates=aspp_rates)

        # decoder: project low-level features to 48ch, concat with ASPP, refine
        self.low_proj = nn.Sequential(nn.Conv2d(256, 48, 1, bias=False),
                                      nn.BatchNorm2d(48), nn.ReLU(inplace=True))
        self.decoder = nn.Sequential(
            SeparableConv2d(256 + 48, 256, 3),
            SeparableConv2d(256, 256, 3))
        self.classifier = nn.Conv2d(256, num_classes, 1)

    def forward(self, x):
        in_size = x.shape[-2:]
        x0 = self.layer0(x)
        low = self.layer1(x0)                 # output stride 4
        x = self.layer2(low)
        x = self.layer3(x)
        x = self.layer4(x)                    # output stride 16
        x = self.aspp(x)
        x = F.interpolate(x, size=low.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, self.low_proj(low)], dim=1)
        x = self.decoder(x)
        x = self.classifier(x)
        x = F.interpolate(x, size=in_size, mode="bilinear", align_corners=False)
        return x                              # (B, num_classes, H, W) logits


# convenience alias so train.py can do `from model import DeepLab`
DeepLab = DeepLabV3Plus


if __name__ == "__main__":
    # quick shape smoke test
    net = DeepLabV3Plus(num_classes=1, output_stride=16, pretrained=False)
    net.eval()
    with torch.no_grad():
        y = net(torch.randn(2, 3, 256, 256))
    n_params = sum(p.numel() for p in net.parameters())
    print("output:", tuple(y.shape), "| params: %.2fM" % (n_params / 1e6))
    assert tuple(y.shape) == (2, 1, 256, 256)
