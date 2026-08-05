# -*- coding: utf-8 -*-
"""
Bruno et al. (2025) -- "A Dual-stage Deep Learning Framework for Breast Ultrasound
Image Segmentation and Classification" (J. Med. Syst., doi:10.1007/s10916-025-02298-6).

This is a PIPELINE paper, not a single new architecture. Its idea (paper Fig. 1):

    Stage 1 (segmentation):  DeepLabV3+ predicts the lesion mask.
    ROI extraction:          the predicted mask's bounding box crops a lesion ROI.
    Stage 2 (classification): a CNN classifies that ROI as benign vs malignant.

So the reproduction ASSEMBLES ready-made blocks we already reproduced:
  * Stage 1 = our DeepLabV3+ from paper [07] (Chen et al. 2018).
  * Stage 2 = an EfficientNet-B0 from paper [06] (Tan & Le 2019), retrained on ROI crops.
The novelty we reproduce is the *segmentation-guided classification* pipeline and the
ROI extraction that couples the two stages -- defined here.

Deviation from the paper: Bruno use a DeepLabV3+ **ResNet-34** backbone; we reuse our
existing DeepLabV3+ **ResNet-50** stage from [07] (torchvision BasicBlock does not
support the atrous/dilated stride DeepLab needs, so ResNet-50 is the faithful atrous
backbone). The pipeline is otherwise identical. Two-class benign/malignant subset,
grayscale, 256x256, exactly as the paper.

Only the NETWORK PIECES + ROI logic live here; the pipeline training/eval is in train.py.
"""
import os
import importlib.util
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

_HERE = os.path.dirname(os.path.abspath(__file__))
PROJ  = os.path.dirname(os.path.dirname(_HERE))

# path to the DeepLabV3+ [07] reproduction (folder name has spaces -> load by path)
DEEPLAB_DIR  = os.path.join(PROJ, "Models", "Chen-2018-DeepLab V3", "DeepLab V3")
DEEPLAB_MODEL_PY = os.path.join(DEEPLAB_DIR, "model.py")
DEEPLAB_CKPT = os.path.join(DEEPLAB_DIR, "DeepLabV3plus_Chen2018.pt")

CLS_NAMES = ("benign", "malignant")            # Bruno: benign/malignant subset only
LABEL2ID  = {"benign": 0, "malignant": 1}


# ------------------------------------------------------------------ #
# Stage 1: segmentation (reuse DeepLabV3+ from paper [07])
# ------------------------------------------------------------------ #
def _load_deeplab_module():
    spec = importlib.util.spec_from_file_location("deeplab_model", DEEPLAB_MODEL_PY)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def build_seg_model(load_checkpoint=True, device="cpu"):
    """DeepLabV3+ (ResNet-50, OS=16) as the Stage-1 lesion segmenter.
    If a trained [07] checkpoint exists, load it (so we reuse that reproduction)."""
    mod = _load_deeplab_module()
    model = mod.DeepLab(num_classes=1, output_stride=16, pretrained=not (load_checkpoint and os.path.exists(DEEPLAB_CKPT)))
    loaded = False
    if load_checkpoint and os.path.exists(DEEPLAB_CKPT):
        state = torch.load(DEEPLAB_CKPT, map_location=device)
        model.load_state_dict(state)
        loaded = True
    return model.to(device), loaded


# ------------------------------------------------------------------ #
# ROI extraction (the coupling between the two stages -- the paper's core)
# ------------------------------------------------------------------ #
def roi_bbox_from_mask(mask, margin=0.15, min_side=16):
    """Bounding box of the (largest) lesion region in a binary mask, expanded by a
    relative `margin`. Returns (x0, y0, x1, y1) or None if the mask is empty."""
    m = (mask > 0).astype(np.uint8)
    if m.sum() == 0:
        return None
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return None
    # largest connected component (skip background label 0)
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h = (stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP],
                  stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT])
    H, W = m.shape
    mx, my = int(round(margin * w)), int(round(margin * h))
    x0, y0 = max(0, x - mx), max(0, y - my)
    x1, y1 = min(W, x + w + mx), min(H, y + h + my)
    if (x1 - x0) < min_side or (y1 - y0) < min_side:      # too small -> unusable
        return None
    return x0, y0, x1, y1


def crop_roi(gray256, bbox, out_size=256):
    """Crop the ROI from a 256x256 grayscale image and resize to out_size.
    If bbox is None (no lesion found) fall back to the whole image (paper behaviour:
    a missed segmentation still gets classified, just with the full frame)."""
    if bbox is None:
        crop = gray256
    else:
        x0, y0, x1, y1 = bbox
        crop = gray256[y0:y1, x0:x1]
    if crop.size == 0:
        crop = gray256
    return cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_AREA)


# ------------------------------------------------------------------ #
# Stage 2: ROI classifier (EfficientNet-B0 from paper [06], 2-class head)
# ------------------------------------------------------------------ #
class ROIClassifier(nn.Module):
    """EfficientNet-B0 backbone with a fresh benign/malignant head, taking a
    (B, 3, 256, 256) ImageNet-normalised ROI crop (grayscale replicated to 3 ch)."""
    def __init__(self, num_classes=2, pretrained=True, dropout=0.2):
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.net = efficientnet_b0(weights=weights)
        in_features = self.net.classifier[1].in_features
        self.net.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def build_roi_classifier(num_classes=2, pretrained=True, dropout=0.2):
    return ROIClassifier(num_classes=num_classes, pretrained=pretrained, dropout=dropout)


if __name__ == "__main__":
    # smoke test of the two blocks + ROI logic
    clf = build_roi_classifier(pretrained=False)
    y = clf(torch.randn(2, 3, 256, 256))
    print("ROIClassifier out:", tuple(y.shape), "params %.2fM" % (sum(p.numel() for p in clf.parameters())/1e6))
    m = np.zeros((256, 256), np.uint8); m[80:150, 100:180] = 1
    bb = roi_bbox_from_mask(m); print("bbox:", bb, "crop:", crop_roi(np.zeros((256, 256), np.uint8), bb).shape)
