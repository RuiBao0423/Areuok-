# -*- coding: utf-8 -*-
"""
Models/ImprovedDualStage/train_from_cache.py

Train the dual-stage Stage-2 classifier from the cached OOF masks
(make_oof_masks.py) -- fast (no segmenter training). Supports:
  * selectable feature set:  --features none | shape | radiomics
  * multi-seed ENSEMBLE:     --seeds 5   (trains N classifiers, averages softmax; also
                             reports the per-seed mean +/- std = training variance)
  * bootstrap 95% CI on the 98-image test set (addresses "is the gap real on so few images")

Every classifier is EfficientNet-B0 (+ optional feature-fusion head), trained on GT-mask
+ OOF-predicted-mask items (train/test distribution match), selected on the val split,
evaluated on test with hflip TTA. All feature values come from the PREDICTED mask at test.

Writes Models/results/<NAME>.json  (NAME from --name, default by feature set).
Examples:
    python Models/ImprovedDualStage/train_from_cache.py --features shape --seeds 1
    python Models/ImprovedDualStage/train_from_cache.py --features radiomics --seeds 5
"""
import os, sys, time, argparse
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from skimage.feature import graycomatrix, graycoprops
from scipy.stats import skew

_HERE = os.path.dirname(os.path.abspath(__file__))
PROJ  = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, PROJ)
sys.path.insert(0, _HERE)
from Models.common import data as D
from Models.common import metrics as M
from Models.common import runner as R
from model import roi_bbox_from_mask, crop_roi, CLS_NAMES
from train import _aug

CACHE = os.path.join(_HERE, "oof_cache.npz")
EPOCHS, BATCH, LR, ROI_SZ = 60, 16, 1e-4, 256


# --------------------------- feature extractors --------------------------- #
def shape_only(mask, gray):
    m = (mask > 0).astype(np.uint8)
    if m.sum() == 0:
        return np.zeros(6, np.float32)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    comp = (lab == k).astype(np.uint8)
    area = float(stats[k, cv2.CC_STAT_AREA])
    x, y, w, h = (stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP],
                  stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT])
    cnt = max(cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0],
              key=cv2.contourArea)
    per = cv2.arcLength(cnt, True)
    circ = float(4 * np.pi * area / (per * per)) if per > 0 else 0.0
    hull = cv2.contourArea(cv2.convexHull(cnt))
    sol = float(area / hull) if hull > 0 else 0.0
    ext = float(area / (w * h)) if w * h > 0 else 0.0
    asp = float(w / h) if h > 0 else 0.0
    ecc = 0.0
    if len(cnt) >= 5:
        (_, _), (MA, ma), _ = cv2.fitEllipse(cnt)
        a, b = max(MA, ma) / 2, min(MA, ma) / 2
        ecc = float(np.sqrt(max(0.0, 1 - b * b / (a * a)))) if a > 0 else 0.0
    return np.array([area / (256 * 256), min(circ, 1.5), sol, ext, asp, ecc], np.float32)


def radiomics(mask, gray):
    shp = shape_only(mask, gray)
    m = mask > 0
    if m.sum() < 10:
        return np.concatenate([shp, np.zeros(8, np.float32)]).astype(np.float32)
    px = gray[m].astype(np.float32); sd = float(px.std())
    inten = np.array([px.mean() / 255, sd / 255,
                      float(skew(px)) if sd > 1e-6 else 0.0], np.float32)
    ys, xs = np.where(m); roi = gray[ys.min():ys.max()+1, xs.min():xs.max()+1]
    q = np.clip(roi.astype(np.int32) // 32, 0, 7).astype(np.uint8)
    glcm = graycomatrix(q, [1], [0, np.pi/2], levels=8, symmetric=True, normed=True)
    tex = np.array([float(graycoprops(glcm, p).mean()) for p in
                    ("contrast", "dissimilarity", "homogeneity", "energy", "correlation")], np.float32)
    return np.concatenate([shp, inten, tex]).astype(np.float32)

FEATS = {"none": (lambda m, g: np.zeros(0, np.float32), 0),
         "shape": (shape_only, 6), "radiomics": (radiomics, 14)}


# --------------------------- model + dataset --------------------------- #
class Classifier(nn.Module):
    def __init__(self, n_feat, num_classes=2, pretrained=True):
        super().__init__()
        self.net = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        in_f = self.net.classifier[1].in_features
        self.net.classifier = nn.Identity()
        self.n_feat = n_feat
        if n_feat > 0:
            self.fmlp = nn.Sequential(nn.BatchNorm1d(n_feat), nn.Linear(n_feat, 32),
                                      nn.ReLU(True), nn.Linear(32, 32), nn.ReLU(True))
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(in_f + (32 if n_feat else 0), num_classes))

    def forward(self, x, feat):
        f = self.net(x)
        if self.n_feat > 0:
            f = torch.cat([f, self.fmlp(feat)], 1)
        return self.head(f)


class DS(Dataset):
    def __init__(self, items, augment=False):
        self.items = items; self.aug = _aug() if augment else None
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        g, feat, y = self.items[i]
        if self.aug is not None:
            g = self.aug(image=g)["image"]
        return D._norm_to_tensor3(g), torch.tensor(feat, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


def build_items(gray, mask, label, featfn):
    out = []
    for i in range(len(label)):
        g = gray[i]; mk = mask[i]
        out.append((crop_roi(g, roi_bbox_from_mask(mk), ROI_SZ), featfn(mk, g), int(label[i])))
    return out


@torch.no_grad()
def proba(clf, items):
    clf.eval(); ld = DataLoader(DS(items), BATCH, shuffle=False); P = []
    for x, f, _ in ld:
        x, f = x.to(R.DEVICE), f.to(R.DEVICE)
        p = torch.softmax(clf(x, f), 1) + torch.softmax(clf(torch.flip(x, [-1]), f), 1)
        P.append((p / 2).cpu().numpy())
    return np.concatenate(P)


def train_one(tr_items, va_items, n_feat, weight, seed):
    R.set_seed(seed)
    clf = Classifier(n_feat).to(R.DEVICE)
    opt = torch.optim.Adam(clf.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    crit = nn.CrossEntropyLoss(weight=weight)
    tr = DataLoader(DS(tr_items, augment=True), BATCH, shuffle=True)
    from sklearn.metrics import f1_score
    best, best_state = -1.0, None
    yv = np.array([y for _, _, y in va_items])
    for ep in range(EPOCHS):
        clf.train()
        for x, f, y in tr:
            x, f, y = x.to(R.DEVICE), f.to(R.DEVICE), y.to(R.DEVICE)
            opt.zero_grad(); crit(clf(x, f), y).backward(); opt.step()
        sched.step()
        pv = proba(clf, va_items).argmax(1)
        vf1 = f1_score(yv, pv, average="macro", labels=[0, 1], zero_division=0)
        if vf1 > best:
            best = vf1; best_state = {k: v.detach().cpu().clone() for k, v in clf.state_dict().items()}
    clf.load_state_dict(best_state)
    return clf, best


def bootstrap_ci(y_true, y_proba, n=2000, seed=0):
    from sklearn.metrics import f1_score
    rng = np.random.RandomState(seed); N = len(y_true); vals = []
    pred = y_proba.argmax(1)
    for _ in range(n):
        idx = rng.randint(0, N, N)
        vals.append(f1_score(y_true[idx], pred[idx], average="macro", labels=[0, 1], zero_division=0))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="radiomics", choices=list(FEATS))
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    featfn, n_feat = FEATS[args.features]
    name = args.name or f"DualStage_OOF_{args.features}" + (f"_ens{args.seeds}" if args.seeds > 1 else "")

    assert os.path.exists(CACHE), "run make_oof_masks.py first"
    z = np.load(CACHE)
    # train items: GT masks + OOF-predicted masks (distribution match)
    tr_items = (build_items(z["tr_gray"], z["tr_gt"], z["tr_label"], featfn) +
                build_items(z["tr_gray"], z["tr_pred"], z["tr_label"], featfn))
    va_items = build_items(z["va_gray"], z["va_pred"], z["va_label"], featfn)
    te_items = build_items(z["te_gray"], z["te_pred"], z["te_label"], featfn)   # pipeline
    gt_items = build_items(z["te_gray"], z["te_gt"], z["te_label"], featfn)     # upper bound
    yte = z["te_label"]

    seg = M.aggregate_seg(list(z["te_pred"]), list(z["te_gt"]))                 # Stage-1 quality
    cnt = np.bincount([y for _, _, y in build_items(z["tr_gray"], z["tr_gt"], z["tr_label"], featfn)],
                      minlength=2).astype(float)
    weight = torch.tensor(cnt.sum() / (2 * np.clip(cnt, 1, None)), dtype=torch.float32, device=R.DEVICE)

    print(f"[{name}] features={args.features} (dim {n_feat})  seeds={args.seeds}  "
          f"train {len(tr_items)}  val {len(va_items)}  test {len(te_items)}")
    t0 = time.time()
    probas, gtprobas, seed_f1 = [], [], []
    from sklearn.metrics import f1_score
    for s in range(args.seeds):
        clf, vbest = train_one(tr_items, va_items, n_feat, weight, seed=42 + s)
        pp = proba(clf, te_items); probas.append(pp); gtprobas.append(proba(clf, gt_items))
        f1s = f1_score(yte, pp.argmax(1), average="macro", labels=[0, 1], zero_division=0)
        seed_f1.append(f1s)
        print(f"  seed {s}: val_bestF1={vbest:.4f}  test_macroF1={f1s:.4f}")
        del clf; torch.cuda.empty_cache()
    train_min = (time.time() - t0) / 60

    ens = np.mean(probas, 0); ens_gt = np.mean(gtprobas, 0)
    cls_pipe = M.classification_metrics(yte, ens.argmax(1), ens, CLS_NAMES)
    cls_gt = M.classification_metrics(yte, ens_gt.argmax(1), ens_gt, CLS_NAMES)
    lo, hi = bootstrap_ci(yte, ens)

    R.plot_confusion(cls_pipe["confusion_matrix"], CLS_NAMES, name)
    result = {
        "model": f"Dual-stage (OOF masks) + {args.features} features"
                 + (f", {args.seeds}-seed ensemble" if args.seeds > 1 else ""),
        "paper": "Our final dual-stage: OOF cross-fitting + radiomics fusion + ensemble",
        "task": "segmentation-guided benign/malignant classification (2-class)",
        "config": {"features": args.features, "n_features": n_feat, "seeds": args.seeds,
                   "roi_size": ROI_SZ, "epochs": EPOCHS, "stage1": "Improved-UNet (OOF train masks)",
                   "stage2": "EfficientNet-B0 + feature fusion", "classes": list(CLS_NAMES)},
        "split": {"train": int(len(z["tr_label"])), "val": int(len(z["va_label"])), "test": int(len(z["te_label"]))},
        "seed_test_macro_f1": [round(v, 4) for v in seed_f1],
        "seed_mean_std": [round(float(np.mean(seed_f1)), 4), round(float(np.std(seed_f1)), 4)],
        "test_macro_f1_bootstrap_ci95": [round(lo, 4), round(hi, 4)],
        "train_minutes": round(train_min, 2),
        "test_segmentation": seg,
        "test_classification": cls_pipe,
        "test_classification_gt_roi": cls_gt,
    }
    R.save_results(name, result)
    print(f"[{name}] ENSEMBLE test macroF1={cls_pipe['macro_f1']:.4f}  acc={cls_pipe['accuracy']:.4f}  "
          f"AUC={cls_pipe['macro_auc_ovr']}  | per-seed {np.mean(seed_f1):.4f}±{np.std(seed_f1):.4f}  "
          f"| 95% CI [{lo:.3f},{hi:.3f}]  | GT-ROI {cls_gt['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
