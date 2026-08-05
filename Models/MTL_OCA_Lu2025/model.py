# -*- coding: utf-8 -*-
"""
Models/MTL_OCA_Lu2025/model.py

Reproduction of MTL-OCA (Lu et al. 2025, Frontiers in Oncology 15:1567577),
"joint segmentation and classification of breast ultrasound via multi-task learning
with Object Contextual Attention".

The paper's primary dataset is called "OASBUD" but its signature (780 images, 600
women, ages 25-75, benign 437 / malignant 210 / normal 133) is EXACTLY BUSI -- so we
reproduce it on BUSI, which is what they actually used.

Architecture (Figure 1-2 of the paper):
  * shared Res-UNet backbone (residual blocks, GroupNorm, ReLU)  -> feeds BOTH tasks
  * segmentation branch = OCA (Object Contextual Attention, an OCR-style module,
    K=2 soft object regions = lesion/background):
        - a coarse 1x1-conv head produces the soft object regions (supervised: L_soft)
        - OCA aggregates region features, computes pixel<->region attention, and
          augments each pixel with its object context -> final mask (supervised: L_aug)
  * classification branch = global-average-pool the backbone bottleneck features ->
    2-layer MLP -> 3-way head (normal / benign / malignant)

Combined loss (train.py):  L = 0.4 * CE(soft_mask) + CE(final_mask) + CE(cls)

Simplifications vs. paper (see README.md): 256x256 input (paper 128); the paper's
"Flatten -> MLP" classification head is replaced by Global-Average-Pool -> MLP
(Flatten on full-res maps is memory-infeasible); standard OCR conv internals for the
unspecified transforms; input = [grayscale, Gaussian-derivative edge magnitude] 2ch.
"""
import os, sys
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

_HERE = os.path.dirname(os.path.abspath(__file__))
PROJ  = os.path.dirname(os.path.dirname(_HERE))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)
from Models.common import data as D                      # letterbox, load_gray_and_mask
import albumentations as A


# --------------------------------------------------------------------------- #
# Res-UNet backbone
# --------------------------------------------------------------------------- #
def _gn(c):
    return nn.GroupNorm(min(8, c), c)


class ResBlock(nn.Module):
    """Conv-GN-ReLU x2 with a residual connection (Res-UNet block)."""
    def __init__(self, cin, cout):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1, bias=False)
        self.n1 = _gn(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.n2 = _gn(cout)
        self.skip = (nn.Identity() if cin == cout
                     else nn.Conv2d(cin, cout, 1, bias=False))
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        r = self.skip(x)
        x = self.act(self.n1(self.conv1(x)))
        x = self.n2(self.conv2(x))
        return self.act(x + r)


class ResUNet(nn.Module):
    """Shared Res-UNet: returns full-res decoder features + the bottleneck features."""
    def __init__(self, in_ch=2, feats=(64, 128, 256, 512)):
        super().__init__()
        self.enc = nn.ModuleList()
        cin = in_ch
        for f in feats:
            self.enc.append(ResBlock(cin, f)); cin = f
        self.pool = nn.MaxPool2d(2)
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for i in range(len(feats) - 1, 0, -1):
            self.up.append(nn.ConvTranspose2d(feats[i], feats[i - 1], 2, stride=2))
            self.dec.append(ResBlock(feats[i - 1] * 2, feats[i - 1]))
        self.out_ch = feats[0]
        self.bottleneck_ch = feats[-1]

    def forward(self, x):
        skips = []
        for i, blk in enumerate(self.enc):
            x = blk(x)
            if i < len(self.enc) - 1:
                skips.append(x)
                x = self.pool(x)
        bottleneck = x
        for j in range(len(self.up)):
            x = self.up[j](x)
            x = torch.cat([x, skips[-1 - j]], dim=1)
            x = self.dec[j](x)
        return x, bottleneck                             # (N,64,H,W), (N,512,H/8,W/8)


# --------------------------------------------------------------------------- #
# OCA (Object Contextual Attention) -- OCR-style object-contextual representations
# --------------------------------------------------------------------------- #
class OCA(nn.Module):
    """K=2 object regions (lesion/background). Produces a coarse soft-region logit
    map (for L_soft) and the augmented, context-refined final logits (for L_aug).

    OCR-style object-contextual attention is designed for LOW-resolution feature maps
    (the original OCRNet runs it at 1/8-1/16 res). We therefore run the pixel<->region
    attention at a fixed small working resolution `work` (default 64) and upsample the
    logits back to full resolution -- this keeps the H*W attention cheap and avoids the
    near-VRAM-limit allocator thrashing that a full-256 attention caused."""
    def __init__(self, in_ch, key=64, K=2, work=64):
        super().__init__()
        self.K = K
        self.work = work
        self.coarse = nn.Conv2d(in_ch, K, 1)             # soft object regions d_k
        self.f_pixel = nn.Sequential(nn.Conv2d(in_ch, key, 1, bias=False), _gn(key),
                                     nn.ReLU(inplace=True))
        self.f_object = nn.Sequential(nn.Conv2d(in_ch, key, 1, bias=False), _gn(key),
                                      nn.ReLU(inplace=True))
        self.f_down = nn.Sequential(nn.Conv2d(in_ch, key, 1, bias=False), _gn(key),
                                    nn.ReLU(inplace=True))
        self.f_up = nn.Sequential(nn.Conv2d(key, in_ch, 1, bias=False), _gn(in_ch),
                                  nn.ReLU(inplace=True))
        self.final = nn.Conv2d(in_ch * 2, K, 1)          # augmented -> final mask logits
        self.key = key

    def forward(self, x):
        H0, W0 = x.shape[-2:]
        # run the attention at a small working resolution, then upsample logits back
        if H0 > self.work:
            x = F.interpolate(x, size=(self.work, self.work), mode="bilinear",
                              align_corners=False)
        N, C, H, W = x.shape
        coarse = self.coarse(x)                          # (N,K,H,W) soft-region logits
        probs = torch.softmax(coarse.view(N, self.K, -1), dim=2)     # (N,K,HW) over pixels
        feats = x.view(N, C, -1).permute(0, 2, 1)        # (N,HW,C)
        region = torch.matmul(probs, feats)              # (N,K,C)  f_k = sum_i d_ki x_i
        region = region.permute(0, 2, 1).unsqueeze(-1)   # (N,C,K,1) for 1x1 convs

        query = self.f_pixel(x).view(N, self.key, -1).permute(0, 2, 1)   # (N,HW,key)
        keym  = self.f_object(region).view(N, self.key, self.K)         # (N,key,K)
        sim = torch.matmul(query, keym) * (self.key ** -0.5)            # (N,HW,K)
        w = torch.softmax(sim, dim=-1)                                   # pixel<->region attn
        value = self.f_down(region).view(N, self.key, self.K).permute(0, 2, 1)  # (N,K,key)
        context = torch.matmul(w, value).permute(0, 2, 1).view(N, self.key, H, W)
        context = self.f_up(context)                                    # (N,C,H,W)
        aug = torch.cat([x, context], dim=1)             # augmented representation
        final = self.final(aug)                          # (N,K,H,W) final mask logits
        if (H, W) != (H0, W0):                           # upsample logits to full res
            coarse = F.interpolate(coarse, size=(H0, W0), mode="bilinear", align_corners=False)
            final  = F.interpolate(final,  size=(H0, W0), mode="bilinear", align_corners=False)
        return coarse, final


class MTLOCA(nn.Module):
    """Full multi-task model: shared Res-UNet -> {OCA seg head, cls MLP head}."""
    def __init__(self, in_ch=2, n_cls=3, K=2):
        super().__init__()
        self.backbone = ResUNet(in_ch=in_ch)
        self.oca = OCA(self.backbone.out_ch, key=64, K=K)
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(self.backbone.bottleneck_ch, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.3), nn.Linear(256, n_cls),
        )

    def forward(self, x):
        feat, bottleneck = self.backbone(x)
        coarse, final = self.oca(feat)                   # seg: (N,2,H,W) each
        logits = self.cls_head(bottleneck)               # cls: (N,n_cls)
        return coarse, final, logits


def count_params_m(model):
    return round(sum(p.numel() for p in model.parameters()) / 1e6, 3)


# --------------------------------------------------------------------------- #
# Multi-task dataset: yields (2ch image, binary mask, class label) for all 3 classes
# --------------------------------------------------------------------------- #
def _edge_channel(gray_u8):
    """Gaussian-derivative (Sobel) magnitude, normalised to uint8 -- the paper's 2nd
    input channel (Eqs. 6-9)."""
    gx = cv2.Sobel(gray_u8, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_u8, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    mx = mag.max()
    mag = (mag / mx * 255.0) if mx > 0 else mag
    return mag.astype(np.uint8)


def _to_tensor2(gray_u8, edge_u8):
    x = np.stack([gray_u8, edge_u8], 0).astype(np.float32) / 255.0
    x = (x - 0.5) / 0.5
    return torch.from_numpy(x)                            # (2,H,W) in [-1,1]


class MTLDataset(Dataset):
    """Letterboxed 256x256 [grayscale, edge] + binary lesion mask + 3-class label.
    normal images have an all-black mask (background only) -- exactly the 3-class
    joint setup of the paper. Preloaded once for fast epochs."""
    def __init__(self, df, split, augment=False):
        rows = df[df.split == split].reset_index(drop=True)
        self.aug = (A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Affine(scale=(0.9, 1.1), translate_percent=0.05, rotate=(-45, 45),
                     border_mode=cv2.BORDER_CONSTANT, p=0.7),
            A.RandomBrightnessContrast(0.2, 0.2, p=0.3),
        ]) if augment else None)
        self.cache = []                                  # (gray_u8, mask_u8, label)
        for _, row in rows.iterrows():
            g, m = D.load_gray_and_mask(row)
            g, m, _ = D.letterbox(g, m)
            self.cache.append((g, (m > 0).astype(np.uint8), int(row["label"])))

    def __len__(self):
        return len(self.cache)

    def __getitem__(self, i):
        g, m, y = self.cache[i]
        if self.aug is not None:
            out = self.aug(image=g, mask=m)
            g, m = out["image"], out["mask"]
        edge = _edge_channel(g)
        x = _to_tensor2(g, edge)
        mask = torch.from_numpy((m > 0).astype(np.int64))          # (H,W) long {0,1}
        return x, mask, torch.tensor(y, dtype=torch.long)
