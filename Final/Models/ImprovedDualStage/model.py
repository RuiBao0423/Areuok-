# -*- coding: utf-8 -*-
"""
Models/ImprovedDualStage/model.py

IMPROVED dual-stage pipeline = our own extension of Bruno et al. (2025) [17].

Bruno's pipeline is  Stage-1 (DeepLabV3+)  ->  ROI crop  ->  Stage-2 (EfficientNet-B0).
Here we swap Stage-1 for our stronger **Improved Pretrained-UNet** segmenter
(EfficientNet-B4 encoder + Focal Tversky + TTA + largest-CC post-processing), which
scored Dice 0.759 vs DeepLabV3+ 0.739. A better Stage-1 mask -> a better ROI crop ->
(hopefully) a better Stage-2 classification. Everything else (ROI extraction, Stage-2
EfficientNet-B0 classifier, leakage-free split, metrics) is reused UNCHANGED from
Bruno so the two pipelines are directly comparable.

Only the network pieces + Stage-1 loader live here; the pipeline train/eval is in train.py.
"""
import os
import importlib.util
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
PROJ  = os.path.dirname(os.path.dirname(_HERE))

# --- reuse the Improved segmenter [our improvement] as Stage 1 -------------------
from Models.Improvement.model import build_model as build_improved_unet
from Models.Improvement.model import predict_mask, predict_prob_tta   # noqa: F401

IMPROVED_CKPT = os.path.join(PROJ, "Models", "Improvement", "improved_unet_best.pt")

# --- reuse Bruno's ROI coupling + Stage-2 classifier UNCHANGED (load by path) ----
_BRUNO_MODEL_PY = os.path.join(PROJ, "Models", "Bruno2025_DualStage", "model.py")


def _load_bruno_module():
    spec = importlib.util.spec_from_file_location("bruno_model", _BRUNO_MODEL_PY)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


_bruno = _load_bruno_module()
roi_bbox_from_mask = _bruno.roi_bbox_from_mask     # ROI extraction (identical to Bruno)
crop_roi           = _bruno.crop_roi
build_roi_classifier = _bruno.build_roi_classifier # Stage-2 EfficientNet-B0 (identical)
CLS_NAMES = _bruno.CLS_NAMES                        # ("benign", "malignant")
LABEL2ID  = _bruno.LABEL2ID


# ------------------------------------------------------------------ #
# Stage 1: segmentation (our Improved Pretrained-UNet, reused checkpoint)
# ------------------------------------------------------------------ #
def build_seg_model(load_checkpoint=True, device="cpu"):
    """Improved Pretrained-UNet (EfficientNet-B4 encoder) as Stage-1 segmenter.
    Loads the checkpoint saved by Models/Improvement/train.py if present."""
    encoder, arch = "efficientnet-b4", "unet"
    have = load_checkpoint and os.path.exists(IMPROVED_CKPT)
    if have:
        ckpt = torch.load(IMPROVED_CKPT, map_location=device)
        encoder = ckpt.get("encoder", encoder); arch = ckpt.get("arch", arch)
    model = build_improved_unet(encoder=encoder, pretrained=not have, arch=arch)
    loaded = False
    if have:
        model.load_state_dict(ckpt["state_dict"])
        loaded = True
    return model.to(device), loaded


if __name__ == "__main__":
    m, ok = build_seg_model(load_checkpoint=False, device="cpu")
    print("Improved Stage-1 built; params %.1fM" % (sum(p.numel() for p in m.parameters())/1e6))
    print("ROI/classifier reused from Bruno:", CLS_NAMES)
