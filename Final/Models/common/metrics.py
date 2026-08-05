# -*- coding: utf-8 -*-
"""
Models/common/metrics.py
Evaluation metrics shared by the baselines (see Evaluation_Metrics.md).

Segmentation (both models): Dice, IoU, pixel accuracy, pixel precision/recall.
Detection (Yap [02] native metric): True Positive Fraction (TPF), False Positives
per image (FPs/image), and F-measure -- computed from the predicted mask by taking
connected-component centroids as detection seed points (Yap et al. 2018, eq. 11-13).
Classification (ResNet [04] etc.): macro-F1 (primary, imbalance-fair), per-class
precision/recall/F1, balanced accuracy, accuracy, macro one-vs-rest ROC-AUC, and the
confusion matrix -- exactly the set specified in Evaluation_Metrics.md.
"""
import numpy as np
import cv2

EPS = 1e-6


# ------------------------- segmentation ------------------------- #
def dice(pred, gt):
    pred = pred.astype(bool); gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    return (2*inter + EPS) / (pred.sum() + gt.sum() + EPS)

def iou(pred, gt):
    pred = pred.astype(bool); gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return (inter + EPS) / (union + EPS)

def pixel_scores(pred, gt):
    pred = pred.astype(bool); gt = gt.astype(bool)
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    acc = (pred == gt).mean()
    prec = (tp + EPS) / (tp + fp + EPS)
    rec  = (tp + EPS) / (tp + fn + EPS)
    return acc, prec, rec

def aggregate_seg(preds, gts):
    """Per-image metrics averaged over the set (mean +/- std)."""
    ds  = np.array([dice(p, g) for p, g in zip(preds, gts)])
    js  = np.array([iou(p, g)  for p, g in zip(preds, gts)])
    ps  = np.array([pixel_scores(p, g) for p, g in zip(preds, gts)])
    acc, prec, rec = ps[:, 0], ps[:, 1], ps[:, 2]
    def ms(a): return dict(mean=round(float(a.mean()), 4), std=round(float(a.std()), 4))
    return {"n": len(ds), "dice": ms(ds), "iou": ms(js),
            "pixel_acc": ms(acc), "pixel_precision": ms(prec), "pixel_recall": ms(rec)}


# ------------------------- detection (Yap) ------------------------- #
def region_centroids(mask, min_area=20):
    """Connected-component centroids of a binary mask (drop tiny regions)."""
    mask = (mask > 0).astype(np.uint8)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out.append((cent[i][0], cent[i][1]))   # (x, y)
    return out

def gt_bbox(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return xs.min(), ys.min(), xs.max(), ys.max()

def detection_metrics(preds, gts, min_area=20):
    """Yap et al. 2018: a detection point (centroid of a predicted region) is a TP if
    it falls inside the ground-truth lesion bounding box, else a FP.
    TPF = detected lesions / actual lesions ; FPs/image ; F-measure = 2TP/(2TP+FP+FN)."""
    TP = FP = FN = 0
    n_img = len(preds)
    for pred, gt in zip(preds, gts):
        box = gt_bbox(gt)
        cents = region_centroids(pred, min_area)
        if box is None:                       # no lesion in GT (shouldn't happen here)
            FP += len(cents); continue
        x0, y0, x1, y1 = box
        hit = [ (x0 <= cx <= x1 and y0 <= cy <= y1) for (cx, cy) in cents ]
        if any(hit):
            TP += 1                           # lesion detected
            FP += sum(1 for h in hit if not h)
        else:
            FN += 1                           # lesion missed
            FP += len(cents)
    actual = TP + FN
    tpf = (TP + EPS) / (actual + EPS)
    fps_img = FP / max(n_img, 1)
    fmeasure = (2*TP + EPS) / (2*TP + FP + FN + EPS)
    return {"TP": int(TP), "FP": int(FP), "FN": int(FN),
            "TPF": round(float(tpf), 4),
            "FPs_per_image": round(float(fps_img), 4),
            "F_measure": round(float(fmeasure), 4)}


# ------------------------- classification ------------------------- #
def classification_metrics(y_true, y_pred, y_proba, class_names):
    """Full classification report (Evaluation_Metrics.md). Primary metric is macro-F1;
    we also report per-class precision/recall/F1, balanced accuracy, accuracy, the
    macro one-vs-rest ROC-AUC (threshold-independent) and the confusion matrix.
      y_true : (N,) int labels
      y_pred : (N,) int predicted labels
      y_proba: (N, C) softmax probabilities (rows sum to 1)
    """
    from sklearn.metrics import (f1_score, roc_auc_score, balanced_accuracy_score,
                                 accuracy_score, confusion_matrix,
                                 precision_recall_fscore_support)
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred); y_proba = np.asarray(y_proba)
    labels = list(range(len(class_names)))

    p, r, f, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    per_class = {class_names[i]: {"precision": round(float(p[i]), 4),
                                  "recall":    round(float(r[i]), 4),
                                  "f1":        round(float(f[i]), 4),
                                  "support":   int(sup[i])} for i in labels}
    try:                                              # AUC needs every class present in y_true
        if len(class_names) == 2:                     # binary: score the positive class column
            macro_auc = round(float(roc_auc_score(y_true, y_proba[:, 1])), 4)
        else:
            macro_auc = round(float(roc_auc_score(
                y_true, y_proba, multi_class="ovr", average="macro", labels=labels)), 4)
    except ValueError:
        macro_auc = None
    return {
        "n": int(len(y_true)),
        "macro_f1":          round(float(f1_score(y_true, y_pred, average="macro",
                                                  labels=labels, zero_division=0)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "accuracy":          round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_auc_ovr":     macro_auc,
        "per_class":         per_class,
        "confusion_matrix":  confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "class_order":       list(class_names),
    }
