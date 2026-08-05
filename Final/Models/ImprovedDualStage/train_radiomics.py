# -*- coding: utf-8 -*-
"""
Models/ImprovedDualStage/train_radiomics.py

RADIOMICS fusion: extends the shape-aware Stage-2 with TEXTURE + INTENSITY features,
so the classifier fuses EfficientNet-B0 deep features with a 14-D hand-crafted
radiomics vector:
  * shape (6):     area_ratio, circularity, solidity, extent, bbox_aspect, eccentricity
  * intensity (3): mean, std, skewness of lesion pixels (echogenicity)
  * GLCM texture (5): contrast, dissimilarity, homogeneity, energy, correlation
    (internal heterogeneity -- what radiologists read to separate benign/malignant;
     the classical BUS-CAD features of the Cheng/Xian surveys).

Everything else matches train_shapeaware.py (same Stage-1, GT+predicted-mask training
items, hflip TTA), so the gap vs. ImprovedDualStage_ShapeAware isolates the added
texture/intensity features. All features are computed from the PREDICTED mask at test
time (train/test match).

Writes Models/results/ImprovedDualStage_Radiomics.json.
Run:  python Models/ImprovedDualStage/train_radiomics.py
"""
import os, sys, time
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from skimage.feature import graycomatrix, graycoprops
from scipy.stats import skew

_HERE = os.path.dirname(os.path.abspath(__file__))
PROJ  = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, PROJ)
sys.path.insert(0, _HERE)
from Models.common import data as D
from Models.common import metrics as M
from Models.common import runner as R
from model import build_seg_model, roi_bbox_from_mask, crop_roi, predict_mask, CLS_NAMES, LABEL2ID
from train import _aug, save_roi_examples
from train_shapeaware import (shape_features, ShapeAwareClassifier, ShapeROIDataset, classify_tta)

NAME    = "ImprovedDualStage_Radiomics"
N_FEAT  = 14
EPOCHS  = 60
BATCH   = 16
LR      = 1e-4
ROI_SZ  = 256
CKPT    = os.path.join(_HERE, "roi_classifier_radiomics.pt")


def radiomics_features(mask, gray256):
    """14-D vector: shape(6) + intensity(3) + GLCM texture(5), from the given mask."""
    shp = shape_features(mask)                                # 6
    m = (mask > 0)
    if m.sum() < 10:
        return np.concatenate([shp, np.zeros(8, np.float32)]).astype(np.float32)
    px = gray256[m].astype(np.float32)
    sd = float(px.std())
    inten = np.array([px.mean() / 255.0, sd / 255.0,
                      float(skew(px)) if sd > 1e-6 else 0.0], np.float32)
    ys, xs = np.where(m)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    roi = gray256[y0:y1, x0:x1]
    q = np.clip(roi.astype(np.int32) // 32, 0, 7).astype(np.uint8)    # 8 grey levels
    glcm = graycomatrix(q, distances=[1], angles=[0, np.pi / 2],
                        levels=8, symmetric=True, normed=True)
    tex = np.array([float(graycoprops(glcm, p).mean()) for p in
                    ("contrast", "dissimilarity", "homogeneity", "energy", "correlation")],
                   np.float32)
    return np.concatenate([shp, inten, tex]).astype(np.float32)


def build_gt_items(df, split):
    items = []
    for _, row in df[df.split == split].iterrows():
        g, m = D.load_gray_and_mask(row)
        g256, m256, _ = D.letterbox(g, m)
        items.append((crop_roi(g256, roi_bbox_from_mask(m256), ROI_SZ),
                      radiomics_features(m256, g256), LABEL2ID[row["cls"]]))
    return items


@torch.no_grad()
def build_pred_items(seg_model, df, split, tta=False, want_examples=False):
    seg_model.eval()
    items, examples, gts, preds = [], [], [], []
    for _, row in df[df.split == split].iterrows():
        g, m = D.load_gray_and_mask(row)
        g256, m256, _ = D.letterbox(g, m)
        x = D._norm_to_tensor3(g256).unsqueeze(0).to(R.DEVICE)
        pr = predict_mask(seg_model, x, thresh=0.5, tta=tta, postprocess=True)[0]
        items.append((crop_roi(g256, roi_bbox_from_mask(pr), ROI_SZ),
                      radiomics_features(pr, g256), LABEL2ID[row["cls"]]))
        if want_examples:
            gts.append((m256 > 0).astype(np.uint8)); preds.append(pr)
            if len(examples) < 6:
                examples.append((g256, (m256 > 0).astype(np.uint8), pr))
    return items, examples, gts, preds


def main():
    R.set_seed(42)
    print(f"[{NAME}] device = {R.DEVICE}")
    df = D.make_split()
    seg_model, loaded = build_seg_model(load_checkpoint=True, device=R.DEVICE)
    print(f"[{NAME}] Stage-1 Improved-UNet reused: {loaded}")

    pred_test, examples, gts, preds = build_pred_items(seg_model, df, "test", tta=True, want_examples=True)
    seg = M.aggregate_seg(preds, gts)
    gt_test = build_gt_items(df, "test")
    gt_tr = build_gt_items(df, "train"); pred_tr, *_ = build_pred_items(seg_model, df, "train")
    pred_va, *_ = build_pred_items(seg_model, df, "val")
    tr_items = gt_tr + pred_tr; va_items = pred_va
    print(f"[{NAME}] Stage-2 train items {len(tr_items)}  val {len(va_items)}  (feat dim {N_FEAT})")

    cnt = np.bincount([y for _, _, y in tr_items], minlength=len(CLS_NAMES))
    w = cnt.sum() / (len(CLS_NAMES) * np.clip(cnt, 1, None))
    weight = torch.tensor(w, dtype=torch.float32, device=R.DEVICE)

    clf = ShapeAwareClassifier(n_shape=N_FEAT, num_classes=len(CLS_NAMES), pretrained=True).to(R.DEVICE)
    opt = torch.optim.Adam(clf.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    crit = nn.CrossEntropyLoss(weight=weight)
    tr = DataLoader(ShapeROIDataset(tr_items, augment=True), BATCH, shuffle=True)

    from sklearn.metrics import f1_score
    hist = {"train_loss": [], "val_f1": []}; best = -1.0
    t0 = time.time()
    for ep in range(EPOCHS):
        clf.train(); losses = []
        for x, shp, y in tr:
            x, shp, y = x.to(R.DEVICE), shp.to(R.DEVICE), y.to(R.DEVICE)
            opt.zero_grad(); loss = crit(clf(x, shp), y); loss.backward(); opt.step()
            losses.append(loss.item())
        sched.step()
        yv, pv, _ = classify_tta(clf, va_items)
        vf1 = float(f1_score(yv, pv, average="macro", labels=[0, 1], zero_division=0))
        hist["train_loss"].append(float(np.mean(losses))); hist["val_f1"].append(vf1)
        if vf1 > best:
            best = vf1; torch.save(clf.state_dict(), CKPT)
        if ep % 10 == 0 or ep == EPOCHS - 1:
            print(f"  ep{ep:02d}  loss {np.mean(losses):.4f}  val_macroF1 {vf1:.4f}  (best {best:.4f})")
    train_min = (time.time() - t0) / 60

    clf.load_state_dict(torch.load(CKPT, map_location=R.DEVICE))
    yt, pt, prt = classify_tta(clf, pred_test)
    cls_pipe = M.classification_metrics(yt, pt, prt, CLS_NAMES)
    yg, pg, prg = classify_tta(clf, gt_test)
    cls_gt = M.classification_metrics(yg, pg, prg, CLS_NAMES)

    R.plot_curve(hist, NAME, keys=("train_loss", "val_f1"))
    R.plot_confusion(cls_pipe["confusion_matrix"], CLS_NAMES, NAME)
    save_roi_examples(examples, [(it[0], it[2]) for it in pred_test], NAME)

    results = {
        "model": "Radiomics-fused dual-stage (Improved-UNet -> ROI + shape+texture+intensity -> EfficientNet-B0*)",
        "paper": "Our extension of Bruno [17]: fuse 14-D radiomics (shape/intensity/GLCM texture) into Stage-2",
        "task": "segmentation-guided benign/malignant classification (2-class)",
        "our_work": "Stage-2 EfficientNet-B0 deep features fused with a 14-D radiomics vector "
                    "(6 shape + 3 intensity + 5 GLCM texture) from the PREDICTED mask; trained on "
                    "gt+predicted items, 60 epochs cosine, hflip TTA. Isolates the texture/intensity "
                    "contribution vs. the shape-only fusion.",
        "config": {"img_size": D.IMG_SIZE, "roi_size": ROI_SZ, "epochs": EPOCHS, "batch": BATCH,
                   "lr": LR, "optimizer": "Adam", "scheduler": "cosine", "n_features": N_FEAT,
                   "stage1": "Improved Pretrained-UNet (reused)",
                   "stage2": "EfficientNet-B0 + radiomics fusion (shape+intensity+GLCM)",
                   "train_data": "gt-mask + predicted-mask items", "classes": list(CLS_NAMES)},
        "split": {k: int((df.split == k).sum()) for k in ["train", "val", "test"]},
        "best_val_macro_f1": round(best, 4), "train_minutes": round(train_min, 2),
        "test_segmentation": seg,
        "test_classification": cls_pipe,
        "test_classification_gt_roi": cls_gt,
        "history": hist,
        "baselines": {"strong_cls": 0.817, "shape_aware": 0.840},
    }
    R.save_results(NAME, results)
    print(f"[{NAME}] PIPELINE test  segDice={seg['dice']['mean']}  || cls macroF1={cls_pipe['macro_f1']}  "
          f"acc={cls_pipe['accuracy']}  AUC={cls_pipe['macro_auc_ovr']}  (GT-ROI F1={cls_gt['macro_f1']})")
    print(f"[{NAME}] vs: strong-cls 0.817 | shape-aware 0.840 | radiomics {cls_pipe['macro_f1']}")


if __name__ == "__main__":
    main()
