# -*- coding: utf-8 -*-
"""
Models/CResUNet_Derakhshandeh2024/model.py

Reproduction of CResU-Net (Derakhshandeh & Mahloojifar, 2024), "Modifying the U-Net's
Encoder-Decoder Architecture for Segmentation of Tumors in Breast Ultrasound Images".
Highest reported BUSI Dice in the surveyed literature: Dice 82.88, IoU 77.5.

NOTE: the "C" in CResU-Net is the **Co-Block** (a dense-CONCATENATION residual block),
NOT a contextual/attention module -- the paper has no attention/ASPP/SE at all. The
novelty is the block schedule below + pure Dice loss.

Block schedule (paper Fig. 2, Table): the encoder stages use DIFFERENT block types:
    E1, E2, E4 = Co-Block   ;   E3 = ResNet-identity   ;   E5 = MultiRes block.
  * Co-Block(F):   three 3x3 convs (F, 2F, 4F) with DENSE CONCATENATIONS -> 7F ch
  * ResIdentity(F): 1x1(F) -> 3x3(F) -> 1x1(4F) + identity shortcut  -> 4F ch
  * MultiRes(F):   three 3x3 convs (F,2F,4F) concatenated + 1x1 residual  -> 7F ch
Loss: pure Dice (paper). Input (N,1,256,256) -> (N,1,256,256) logits.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _cbr(cin, cout, k=3, pad=1):
    return nn.Sequential(nn.Conv2d(cin, cout, k, padding=pad, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class CoBlock(nn.Module):
    """Dense-concatenation residual block. Output = 7F channels."""
    def __init__(self, cin, F_):
        super().__init__()
        self.c1 = _cbr(cin, F_)
        self.c2 = _cbr(F_, 2 * F_)
        self.c3 = _cbr(3 * F_, 4 * F_)
        self.out_ch = 7 * F_

    def forward(self, x):
        x1 = self.c1(x)
        x2 = self.c2(x1)
        c1 = torch.cat([x2, x1], dim=1)                  # 3F
        x3 = self.c3(c1)
        return torch.cat([x3, c1], dim=1)                # 7F


class ResIdentity(nn.Module):
    """ResNet bottleneck identity block. Output = 4F channels."""
    def __init__(self, cin, F_):
        super().__init__()
        self.y1 = _cbr(cin, F_, k=1, pad=0)
        self.y2 = _cbr(F_, F_, k=3, pad=1)
        self.y3 = nn.Sequential(nn.Conv2d(F_, 4 * F_, 1, bias=False),
                                nn.BatchNorm2d(4 * F_))
        self.proj = nn.Conv2d(cin, 4 * F_, 1, bias=False)
        self.act = nn.ReLU(inplace=True)
        self.out_ch = 4 * F_

    def forward(self, x):
        s = self.proj(x)
        y = self.y3(self.y2(self.y1(x)))
        return self.act(y + s)


class MultiRes(nn.Module):
    """MultiResUNet-style block: 3 stacked convs concatenated + 1x1 residual. Out = 7F."""
    def __init__(self, cin, F_):
        super().__init__()
        self.a1 = nn.Conv2d(cin, F_, 3, padding=1, bias=False)
        self.a2 = nn.Conv2d(F_, 2 * F_, 3, padding=1, bias=False)
        self.a3 = nn.Conv2d(2 * F_, 4 * F_, 3, padding=1, bias=False)
        self.res = nn.Conv2d(cin, 7 * F_, 1, bias=False)
        self.bn = nn.BatchNorm2d(7 * F_)
        self.act = nn.ReLU(inplace=True)
        self.out_ch = 7 * F_

    def forward(self, x):
        a1 = self.a1(x); a2 = self.a2(a1); a3 = self.a3(a2)
        cat = torch.cat([a1, a2, a3], dim=1)             # 7F
        return self.act(self.bn(cat + self.res(x)))


def _make_block(kind, cin, F_):
    return {"co": CoBlock, "res": ResIdentity, "multi": MultiRes}[kind](cin, F_)


class CResUNet(nn.Module):
    """U-Net with the CResU-Net block schedule. Each block output is 1x1-projected to a
    fixed stage width to keep channel bookkeeping and params in check (the paper gives
    no per-layer channel table)."""
    def __init__(self, in_ch=1, n_classes=1):
        super().__init__()
        Fs    = [16, 32, 64, 128, 256]                   # paper base filter counts
        kinds = ["co", "co", "res", "co", "multi"]       # E1,E2,E3,E4,E5 block types
        widths = [32, 64, 128, 256, 512]                 # stage output widths after 1x1 proj

        self.enc, self.enc_proj = nn.ModuleList(), nn.ModuleList()
        cin = in_ch
        for kind, Ff, w in zip(kinds, Fs, widths):
            blk = _make_block(kind, cin, Ff)
            self.enc.append(blk)
            self.enc_proj.append(nn.Conv2d(blk.out_ch, w, 1))
            cin = w
        self.pool = nn.MaxPool2d(2)
        self.drop = nn.Dropout2d(0.2)

        self.bottleneck = nn.Sequential(_cbr(widths[-1], 512), _cbr(512, 512),
                                        nn.Dropout2d(0.2))

        # decoder: Co-Blocks throughout (documented simplification of the per-stage
        # block variation), single skip; 1x1 project each block to the decoder width.
        self.up, self.dec, self.dec_proj = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        dec_in_prev = 512
        for i in range(len(widths) - 1, -1, -1):
            skip_w = widths[i]
            blk = CoBlock(dec_in_prev + skip_w, skip_w // 2 if skip_w > 16 else 16)
            self.up.append(nn.ConvTranspose2d(dec_in_prev, dec_in_prev, 2, stride=2))
            self.dec.append(blk)
            self.dec_proj.append(nn.Conv2d(blk.out_ch, skip_w, 1))
            dec_in_prev = skip_w
        self.head = nn.Conv2d(widths[0], n_classes, 1)

    def forward(self, x):
        skips = []
        for i in range(len(self.enc)):
            x = self.enc_proj[i](self.enc[i](x))
            if i == len(self.enc) - 1:
                x = self.drop(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for j in range(len(self.dec)):
            x = self.up[j](x)
            skip = skips[-1 - j]
            x = torch.cat([x, skip], dim=1)
            x = self.dec_proj[j](self.dec[j](x))
        return self.head(x)


def count_params_m(model):
    return round(sum(p.numel() for p in model.parameters()) / 1e6, 3)
