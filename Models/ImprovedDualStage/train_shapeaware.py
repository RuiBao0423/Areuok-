# -*- coding: utf-8 -*-
"""
Models/ImprovedDualStage/train_shapeaware.py

SHAPE-AWARE Stage-2 for the improved dual-stage pipeline. This is the strengthened
Stage-2 (train_strong.py) PLUS a fusion of hand-crafted lesion-SHAPE features into the
classifier -- motivated by our own EDA (Section 3.5), which showed that lesion shape
(circularity, solidity, eccentricity, ...) strongly separates benign from malignant,
and by the finding that the dual-stage pipeline is bottlenecked by the classifier.

Key methodological point (train/test distribution match): shape features are computed
from the *predicted* mask at test time (no ground truth is available at deployment), so
the classifier is TRAINED on both ground-truth-mask and predicted-mask shape features
(exactly like the strengthened Stage-2's ROI crops) rather than on clean GT features
only -- otherwise it would over-rely on clean features it never sees at test.

Classifier: EfficientNet-B0 pooled features (1280-d) concatenated with a small MLP over
the 6 shape descriptors, then a 2-way head. Everything else matches train_strong.py, so
the ONLY difference vs. ImprovedDualStage_StrongCls is the shape fusion -> a clean
controlled comparison isolating the shape-feature contribution.

Writes Models/results/ImprovedDualStage_ShapeAware.json.
Run:  python Models/ImprovedDualStage/train_shapeaware.py
"""
import os, sys, time
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

_HERE = os.path.dirname(os.path.abspath(__file__))
PROJ  = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, PROJ)
sys.path.insert(0, _HERE)
from Models.common import data as D
from Models.common import metrics as M
from Models.common import runner as R
from model import build_seg_model, roi_bbox_from_mask, crop_roi, predict_mask, CLS_NAMES, LABEL2ID
from train import _aug, save_roi_examples

NAME   = "ImprovedDualStage_ShapeAware"
EPOCHS = 60
BATCH  = 16
LR     = 1e-4
ROI_SZ = 256
N_SHAPE = 6
CKPT   = os.path.join(_HERE, "roi_classifier_shapeaware.pt")


# --------------------------------------------------------------------------- #
# lesion-shape descriptors from a binary mask (same features validated in EDA 3.5)
# --------------------------------------------------------------------------- #
def shape_features(mask):
    """[area_ratio, circularity, solidity, extent, bbox_aspect, eccentricity].
    Computed from the largest connected component; all zeros if the mask is empty."""
    m = (mask > 0).astype(np.uint8)
    H, W = m.shape
    if m.sum() == 0:
        return np.zeros(N_SHAPE, np.float32)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    comp = (lab == k).astype(np.uint8)
    area = float(stats[k, cv2.CC_STAT_AREA])
    x, y, w, h = (stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP],
                  stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT])
    cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(cnts, key=cv2.contourArea)
    perim = cv2.arcLength(cnt, True)
    circularity = float(4 * np.pi * area / (perim * perim)) if perim > 0 else 0.0
    hull_area = cv2.contourArea(cv2.convexHull(cnt))
    solidity = float(area / hull_area) if hull_area > 0 else 0.0
    extent = float(area / (w * h)) if w * h > 0 else 0.0
    bbox_aspect = float(w / h) if h > 0 else 0.0
    ecc = 0.0
    if len(cnt) >= 5:
        (_, _), (MA, ma), _ = cv2.fitEllipse(cnt)
        a, b = max(MA, ma) / 2.0, min(MA, ma) / 2.0
        ecc = float(np.sqrt(max(0.0, 1 - (b * b) / (a * a)))) if a > 0 else 0.0
    area_ratio = area / (H * W)
    return np.array([area_ratio, min(circularity, 1.5), solidity, extent,
                     bbox_aspect, ecc], np.float32)


# --------------------------------------------------------------------------- #
# shape-aware classifier: EfficientNet-B0 features (+) shape-MLP
# --------------------------------------------------------------------------- #
class ShapeAwareClassifier(nn.Module):
    def __init__(self, n_shape=N_SHAPE, num_classes=2, pretrained=True, dropout=0.2):
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.net = efficientnet_b0(weights=weights)
        in_f = self.net.classifier[1].in_features            # 1280
        self.net.classifier = nn.Identity()                  # -> pooled features (B,1280)
        self.shape_mlp = nn.Sequential(
            nn.BatchNorm1d(n_shape),                          # auto-standardise shape scales
            nn.Linear(n_shape, 32), nn.ReLU(inplace=True),
            nn.Linear(32, 32), nn.ReLU(inplace=True))
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_f + 32, num_classes))

    def forward(self, x, shape):
        f = self.net(x)                                      # (B,1280)
        s = self.shape_mlp(shape)                            # (B,32)
        return self.head(torch.cat([f, s], dim=1))


# --------------------------------------------------------------------------- #
class ShapeROIDataset(Dataset):
    """(ROI crop, shape vector, label). Augmentation perturbs the crop image only;
    shape descriptors are geometric properties of the mask (flip/scale-robust)."""
    def __init__(self, items, augment=False):
        self.items = items
        self.aug = _aug() if augment else None

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        g, shp, y = self.items[i]
        if self.aug is not None:
            g = self.aug(image=g)["image"]
        x = D._norm_to_tensor3(g)
        return x, torch.tensor(shp, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


def build_gt_items(df, split):
    """ROI crop + shape from GROUND-TRUTH mask."""
    items = []
    for _, row in df[df.split == split].iterrows():
        g, m = D.load_gray_and_mask(row)
        g256, m256, _ = D.letterbox(g, m)
        bbox = roi_bbox_from_mask(m256)
        items.append((crop_roi(g256, bbox, ROI_SZ), shape_features(m256), LABEL2ID[row["cls"]]))
    return items


@torch.no_grad()
def build_pred_items(seg_model, df, split, tta=False, want_examples=False):
    """ROI crop + shape from PREDICTED mask (Stage-1)."""
    seg_model.eval()
    items, examples, gts, preds = [], [], [], []
    for _, row in df[df.split == split].iterrows():
        g, m = D.load_gray_and_mask(row)
        g256, m256, _ = D.letterbox(g, m)
        x = D._norm_to_tensor3(g256).unsqueeze(0).to(R.DEVICE)
        pr = predict_mask(seg_model, x, thresh=0.5, tta=tta, postprocess=True)[0]
        items.append((crop_roi(g256, roi_bbox_from_mask(pr), ROI_SZ),
                      shape_features(pr), LABEL2ID[row["cls"]]))
        if want_examples:
            gts.append((m256 > 0).astype(np.uint8)); preds.append(pr)
            if len(examples) < 6:
                examples.append((g256, (m256 > 0).astype(np.uint8), pr))
    return items, examples, gts, preds


@torch.no_grad()
def classify_tta(clf, items):
    """Predict with horizontal-flip TTA. items carry shape vectors."""
    clf.eval()
    ld = DataLoader(ShapeROIDataset(items, augment=False), BATCH, shuffle=False)
    ys, ps, pr = [], [], []
    for x, shp, y in ld:
        x, shp = x.to(R.DEVICE), shp.to(R.DEVICE)
        p = torch.softmax(clf(x, shp), 1) + torch.softmax(clf(torch.flip(x, dims=[-1]), shp), 1)
        p = (p / 2).cpu().numpy()
        ys.append(y.numpy()); ps.append(p.argmax(1)); pr.append(p)
    return np.concatenate(ys), np.concatenate(ps), np.concatenate(pr)


def main():
    R.set_seed(42)
    print(f"[{NAME}] device = {R.DEVICE}")
    df = D.make_split()

    seg_model, loaded = build_seg_model(load_checkpoint=True, device=R.DEVICE)
    print(f"[{NAME}] Stage-1 Improved-UNet reused: {loaded}")

    # test: predicted-mask items (+ examples + seg quality) and GT-mask items (upper bound)
    pred_test, examples, gts, preds = build_pred_items(seg_model, df, "test", tta=True, want_examples=True)
    seg = M.aggregate_seg(preds, gts)
    gt_test = build_gt_items(df, "test")

    # Stage-2 training data: GT-mask + predicted-mask items (distribution match)
    gt_tr = build_gt_items(df, "train"); gt_va = build_gt_items(df, "val")
    pred_tr, *_ = build_pred_items(seg_model, df, "train"); pred_va, *_ = build_pred_items(seg_model, df, "val")
    tr_items = gt_tr + pred_tr
    va_items = pred_va
    print(f"[{NAME}] Stage-2 train items {len(tr_items)} (gt {len(gt_tr)} + pred {len(pred_tr)})  val {len(va_items)}")

    cnt = np.bincount([y for _, _, y in tr_items], minlength=len(CLS_NAMES))
    w = cnt.sum() / (len(CLS_NAMES) * np.clip(cnt, 1, None))
    weight = torch.tensor(w, dtype=torch.float32, device=R.DEVICE)

    clf = ShapeAwareClassifier(num_classes=len(CLS_NAMES), pretrained=True).to(R.DEVICE)
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
        "model": "Shape-aware dual-stage (Improved-UNet -> ROI + predicted-mask shape -> EfficientNet-B0*)",
        "paper": "Our extension of Bruno [17]: fuse EDA-validated lesion-shape features into Stage-2",
        "task": "segmentation-guided benign/malignant classification (2-class)",
        "our_work": "Stage-2 EfficientNet-B0 features fused with 6 lesion-shape descriptors "
                    "(area-ratio, circularity, solidity, extent, bbox-aspect, eccentricity) computed "
                    "from the PREDICTED mask; trained on gt+predicted items (train/test match), "
                    "60 epochs cosine, hflip TTA. Same Stage-1 (Improved-UNet) as the other pipelines.",
        "config": {"img_size": D.IMG_SIZE, "roi_size": ROI_SZ, "epochs": EPOCHS, "batch": BATCH,
                   "lr": LR, "optimizer": "Adam", "scheduler": "cosine", "loss": "weighted CrossEntropy",
                   "stage1": "Improved Pretrained-UNet (reused)",
                   "stage2": "EfficientNet-B0 + shape-feature fusion (6 descriptors)",
                   "shape_features": ["area_ratio", "circularity", "solidity", "extent",
                                      "bbox_aspect", "eccentricity"],
                   "train_data": "gt-mask + predicted-mask items", "classes": list(CLS_NAMES),
                   "stage1_checkpoint_reused": bool(loaded)},
        "split": {k: int((df.split == k).sum()) for k in ["train", "val", "test"]},
        "best_val_macro_f1": round(best, 4), "train_minutes": round(train_min, 2),
        "test_segmentation": seg,
        "test_classification": cls_pipe,
        "test_classification_gt_roi": cls_gt,
        "history": hist,
        "baselines": {"bruno": 0.804, "improved_vanilla": 0.807, "strong_cls": 0.817},
    }
    R.save_results(NAME, results)
    print(f"[{NAME}] PIPELINE test  segDice={seg['dice']['mean']}  || cls macroF1={cls_pipe['macro_f1']}  "
          f"acc={cls_pipe['accuracy']}  AUC={cls_pipe['macro_auc_ovr']}  (GT-ROI F1={cls_gt['macro_f1']})  ({train_min:.1f} min)")
    print(f"[{NAME}] vs: Bruno 0.804 | vanilla 0.807 | strong-cls 0.817 | shape-aware {cls_pipe['macro_f1']}")


if __name__ == "__main__":
    main()
