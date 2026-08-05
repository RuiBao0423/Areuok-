# -*- coding: utf-8 -*-
"""
Models/common/losses.py
Segmentation losses for the BUSI improvement experiments.

Motivation (from the results analysis of the baselines):
  The from-scratch U-Net / Attention U-Net reach a *mean* Dice of ~0.63-0.65 but
  with a very large per-image std (~0.36). A Dice bounded in [0,1] with that
  mean/std can only be BIMODAL: most images score ~0.85 while a sub-population is
  MISSED entirely (Dice ~ 0). Those misses are false-negative-dominated.

  Plain Dice+BCE weights false positives and false negatives symmetrically. To
  attack the missed-lesion tail we use the TVERSKY family, which lets us penalise
  false negatives (missed lesion pixels) more than false positives:

      Tversky      = TP / (TP + alpha*FP + beta*FN)      (alpha<beta -> FN hurt more)
      FocalTversky = (1 - Tversky) ** gamma              (focus on hard/low-overlap cases)

  Refs: Salehi et al. 2017 (Tversky loss); Abraham & Khan 2019 (Focal Tversky).

All losses take raw LOGITS of shape (N,1,H,W) and a float target in {0,1} of the
same shape -- exactly what smp.Unet outputs and what SegDataset returns.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _flatten_probs(logits, target):
    """logits/target (N,1,H,W) -> per-image flattened probs & targets (N, H*W)."""
    prob = torch.sigmoid(logits)
    n = prob.shape[0]
    return prob.reshape(n, -1), target.reshape(n, -1)


class TverskyLoss(nn.Module):
    """Soft Tversky loss. alpha weights FP, beta weights FN (alpha+beta typically 1).
    alpha=beta=0.5 recovers Dice. Use alpha<beta (e.g. 0.3/0.7) to punish misses."""
    def __init__(self, alpha=0.3, beta=0.7, smooth=1.0):
        super().__init__()
        self.alpha, self.beta, self.smooth = alpha, beta, smooth

    def forward(self, logits, target):
        p, g = _flatten_probs(logits, target)
        tp = (p * g).sum(1)
        fp = (p * (1 - g)).sum(1)
        fn = ((1 - p) * g).sum(1)
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return (1 - tversky).mean()


class FocalTverskyLoss(nn.Module):
    """Focal Tversky: (1 - Tversky) ** gamma. gamma>1 focuses training on the
    low-overlap (hard) images -- i.e. exactly the missed-lesion tail that inflates
    the baseline Dice std. Defaults follow Abraham & Khan 2019 (alpha .3/beta .7,
    gamma 4/3)."""
    def __init__(self, alpha=0.3, beta=0.7, gamma=1.3333, smooth=1.0):
        super().__init__()
        self.alpha, self.beta, self.gamma, self.smooth = alpha, beta, gamma, smooth

    def forward(self, logits, target):
        p, g = _flatten_probs(logits, target)
        tp = (p * g).sum(1)
        fp = (p * (1 - g)).sum(1)
        fn = ((1 - p) * g).sum(1)
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return ((1 - tversky) ** self.gamma).mean()


class ComboLoss(nn.Module):
    """Focal Tversky + BCE. The BCE term keeps pixel-level calibration/gradients
    healthy early in training; the Focal Tversky term drives the overlap metric and
    the FN penalty. `bce_w` blends the two. This is the drop-in replacement for the
    baselines' Dice+BCE.

    Set focal=False to get plain (non-focal) Tversky+BCE, or alpha=beta=0.5 to get
    the original symmetric Dice+BCE behaviour (useful as an ablation control)."""
    def __init__(self, alpha=0.3, beta=0.7, gamma=1.3333, bce_w=0.5, focal=True,
                 pos_weight=None):
        super().__init__()
        self.tversky = (FocalTverskyLoss(alpha, beta, gamma) if focal
                        else TverskyLoss(alpha, beta))
        self.bce_w = bce_w
        self.register_buffer("pos_weight",
                             torch.tensor([pos_weight]) if pos_weight else None)

    def forward(self, logits, target):
        pw = self.pos_weight
        bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pw)
        return self.bce_w * bce + (1 - self.bce_w) * self.tversky(logits, target)
