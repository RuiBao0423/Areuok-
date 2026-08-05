# -*- coding: utf-8 -*-
"""
Models/AAUNet_Chen2023/model.py

Reproduction of AAU-Net (Chen et al., IEEE TMI 2022/2023), "An Adaptive Attention
U-Net for Breast Lesions Segmentation in Ultrasound Images". Official ref:
https://github.com/CGPxy/AAU-net . Reported on BUSI (4-fold): Dice 77.51, IoU 68.82.

Core idea: replace every U-Net conv block with the HAAM (Hybrid Adaptive Attention
Module). HAAM has three parts (Fig. 2-3 of the paper):
  (1) three PARALLEL multi-receptive-field convs on the block input:
        F3  = 3x3 conv                     (RF 3)
        F5  = 5x5 conv                      (RF ~5)
        FD  = 3x3 dilated conv, rate 3      (RF ~11)
      each = Conv - BN - LeakyReLU.
  (2) CHANNEL self-attention: from GAP(F5) & GAP(FD) it learns a channel map
        alpha in [0,1]; the COMPLEMENT (1-alpha) adaptively routes the two large-RF
        branches:  FD_C = alpha * FD ,  FS_C = (1-alpha) * F5.
  (3) SPATIAL self-attention: a single-channel map beta in [0,1] (and 1-beta)
        adaptively fuses the location info of the 3x3 branch vs. the channel-refined
        branches -> F_out.

Each encoder/decoder stage = two stacked HAAMs (paper). U-Net skeleton otherwise:
5 levels [32,64,128,256,512], maxpool down, bilinear-upsample + concat skip up.

Interface parity with the seg baselines: input (N,1,256,256) -> (N,1,256,256) logits.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class HAAM(nn.Module):
    """Hybrid Adaptive Attention Module (replaces a conv block)."""
    def __init__(self, cin, cout):
        super().__init__()
        def cbl(k, pad, dil=1):
            return nn.Sequential(
                nn.Conv2d(cin, cout, k, padding=pad, dilation=dil, bias=False),
                nn.BatchNorm2d(cout), nn.LeakyReLU(0.2, inplace=True))
        self.conv3 = cbl(3, 1)
        self.conv5 = cbl(5, 2)
        self.dconv = cbl(3, 3, dil=3)                    # dilated 3x3, rate 3
        # channel self-attention: GAP(F5)||GAP(FD) -> FC(2c->c)->BN->ReLU -> FC(c->c)->sigmoid
        self.ch_fc1 = nn.Sequential(nn.Conv2d(2 * cout, cout, 1, bias=False),
                                    nn.BatchNorm2d(cout), nn.ReLU(inplace=True))
        self.ch_fc2 = nn.Conv2d(cout, cout, 1)
        # spatial self-attention
        self.sp_f3 = nn.Conv2d(cout, cout, 1)
        self.sp_fc = nn.Conv2d(cout, cout, 1)
        self.sp_beta = nn.Conv2d(cout, 1, 1)
        self.out = nn.Conv2d(cout, cout, 1)

    def forward(self, x):
        F3, F5, FD = self.conv3(x), self.conv5(x), self.dconv(x)
        # ---- channel self-attention
        g = torch.cat([F.adaptive_avg_pool2d(F5, 1),
                       F.adaptive_avg_pool2d(FD, 1)], dim=1)         # (B,2C,1,1)
        alpha = torch.sigmoid(self.ch_fc2(self.ch_fc1(g)))          # (B,C,1,1)
        FD_C = alpha * FD
        FS_C = (1 - alpha) * F5
        # ---- spatial self-attention
        FS1 = self.sp_f3(F3)
        FS1_C = self.sp_fc(FS_C + FD_C)
        beta = torch.sigmoid(self.sp_beta(F.relu(FS1 + FS1_C)))     # (B,1,H,W)
        F_out = self.out(beta * FS1_C + (1 - beta) * FS1)
        return F_out


class _Stage(nn.Module):
    """Two stacked HAAMs = one U-Net stage."""
    def __init__(self, cin, cout):
        super().__init__()
        self.h1 = HAAM(cin, cout)
        self.h2 = HAAM(cout, cout)

    def forward(self, x):
        return self.h2(self.h1(x))


class AAUNet(nn.Module):
    def __init__(self, in_ch=1, n_classes=1, feats=(32, 64, 128, 256, 512)):
        super().__init__()
        self.enc = nn.ModuleList()
        cin = in_ch
        for f in feats:
            self.enc.append(_Stage(cin, f)); cin = f
        self.pool = nn.MaxPool2d(2)
        self.dec = nn.ModuleList()
        self.reduce = nn.ModuleList()
        for i in range(len(feats) - 1, 0, -1):
            # bilinear up keeps channels feats[i]; concat skip feats[i-1] -> stage
            self.dec.append(_Stage(feats[i] + feats[i - 1], feats[i - 1]))
        self.head = nn.Conv2d(feats[0], n_classes, 1)

    def forward(self, x):
        skips = []
        for i, stage in enumerate(self.enc):
            x = stage(x)
            if i < len(self.enc) - 1:
                skips.append(x)
                x = self.pool(x)
        for j, stage in enumerate(self.dec):
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            x = torch.cat([x, skips[-1 - j]], dim=1)
            x = stage(x)
        return self.head(x)


def count_params_m(model):
    return round(sum(p.numel() for p in model.parameters()) / 1e6, 3)
