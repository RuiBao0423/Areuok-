# -*- coding: utf-8 -*-
"""
Models/PretrainedUNet_Improved/model.py

IMPROVEMENT over the from-scratch U-Net [03] / Attention U-Net [09] baselines.
Three changes, each targeted at a failure mode measured in the baseline results:

  (1) ImageNet-PRETRAINED encoder (U-Net decoder kept).
      Evidence: the only segmentation baseline that used transfer learning
      (DeepLabV3+, ResNet-50) scored Dice 0.739 vs 0.648 / 0.629 for the
      from-scratch U-Net / Attention U-Net. Transfer learning is worth ~+0.09-0.11
      Dice on this 645-image set. We build a U-Net with an EfficientNet/ResNet
      encoder initialised from ImageNet via `segmentation_models_pytorch`.

  (2) Focal Tversky + BCE loss (see common/losses.py) to attack the FN-dominated
      missed-lesion tail that gives the baselines their large Dice std (~0.36).

  (3) Test-time augmentation (hflip + multi-scale) and largest-connected-component
      post-processing at inference (see `predict_mask`) to remove spurious blobs
      and recover borderline lesions.

Interface parity with the baselines: build a model that maps (N,3,256,256) ->
(N,1,256,256) logits, trained on SegDataset(..., rgb3=True). `predict_mask`
returns a {0,1} uint8 256x256 mask ready for common.metrics.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

try:
    import segmentation_models_pytorch as smp
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "This improvement needs segmentation-models-pytorch:\n"
        "    pip install segmentation-models-pytorch\n" + str(e))


def build_model(encoder="efficientnet-b4", pretrained=True, arch="unet"):
    """U-Net (or U-Net++) with an ImageNet-pretrained encoder.
      encoder    : any smp encoder, e.g. 'efficientnet-b4', 'resnet34', 'timm-efficientnet-b4'
      pretrained : ImageNet weights for the encoder (the transfer-learning part)
      arch       : 'unet' (default) or 'unetpp' (UNet++, denser skips -> small lesions)
    Input: 3-channel ImageNet-normalised (SegDataset rgb3=True). Output: (N,1,H,W) logits.
    """
    weights = "imagenet" if pretrained else None
    Arch = {"unet": smp.Unet, "unetpp": smp.UnetPlusPlus}[arch]
    net = Arch(
        encoder_name=encoder,
        encoder_weights=weights,
        in_channels=3,
        classes=1,                 # binary lesion -> 1 logit channel
        activation=None,           # return raw logits (loss applies sigmoid)
    )
    return net


def count_params_m(model):
    return round(sum(p.numel() for p in model.parameters()) / 1e6, 3)


# --------------------------------------------------------------------------- #
# Inference: test-time augmentation + largest-connected-component clean-up
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _forward_prob(model, x):
    """x (N,3,H,W) -> sigmoid prob (N,1,H,W)."""
    return torch.sigmoid(model(x))


@torch.no_grad()
def predict_prob_tta(model, x, scales=(1.0, 0.75, 1.25), hflip=True):
    """Average sigmoid probabilities over horizontal flip + multiple scales.
    x: (N,3,H,W) tensor already on the model's device. Returns (N,1,H,W) prob."""
    model.eval()
    H, W = x.shape[-2:]
    acc = _forward_prob(model, x)
    count = 1.0
    if hflip:
        acc = acc + torch.flip(_forward_prob(model, torch.flip(x, dims=[-1])), dims=[-1])
        count += 1.0
    for s in scales:
        if abs(s - 1.0) < 1e-6:
            continue
        xs = F.interpolate(x, scale_factor=s, mode="bilinear", align_corners=False)
        ps = _forward_prob(model, xs)
        ps = F.interpolate(ps, size=(H, W), mode="bilinear", align_corners=False)
        acc = acc + ps
        count += 1.0
        if hflip:
            xf = torch.flip(xs, dims=[-1])
            pf = _forward_prob(model, xf)
            pf = torch.flip(pf, dims=[-1])
            pf = F.interpolate(pf, size=(H, W), mode="bilinear", align_corners=False)
            acc = acc + pf
            count += 1.0
    return acc / count


def keep_largest_cc(mask, min_area=30):
    """Keep only the largest connected component (+ any component comparably large).
    Removes the speckle false-positives that hurt pixel-precision."""
    mask = (mask > 0).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    biggest = 1 + int(np.argmax(areas))
    big_area = stats[biggest, cv2.CC_STAT_AREA]
    keep = np.zeros_like(mask)
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        if i == biggest or a >= max(min_area, 0.30 * big_area):  # main lesion + comparably-large
            keep[lab == i] = 1
    return keep


@torch.no_grad()
def predict_mask(model, x, thresh=0.5, tta=True, postprocess=True,
                 close_ksize=5, min_area=30):
    """Full inference for ONE batch. x (N,3,H,W) on device.
    Returns a list of N {0,1} uint8 256x256 masks (numpy), ready for common.metrics.
      thresh      : probability threshold
      tta         : horizontal-flip + multi-scale averaging
      postprocess : morphological close + largest-connected-component
    """
    prob = (predict_prob_tta(model, x) if tta else _forward_prob(model, x))
    prob = prob.squeeze(1).cpu().numpy()               # (N,H,W)
    out = []
    for p in prob:
        m = (p >= thresh).astype(np.uint8)
        if postprocess:
            if close_ksize and close_ksize > 1:
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
                m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
            m = keep_largest_cc(m, min_area=min_area)
        out.append(m.astype(np.uint8))
    return out
