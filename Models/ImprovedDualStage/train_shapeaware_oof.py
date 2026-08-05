# -*- coding: utf-8 -*-
"""
Models/ImprovedDualStage/train_shapeaware_oof.py

Rigorous OUT-OF-FOLD (OOF) version of the shape-aware dual-stage classifier.

Motivation (train/test match, taken to its logical end): in train_shapeaware.py the
predicted masks used to build the *training* items come from the SAME Stage-1 segmenter
that was trained on those images (in-fold), so the training masks are optimistically
clean compared with the test masks. Here we remove that optimism with cross-fitting /
stacking: the training set is split into K folds by near-duplicate cluster, and each
fold's masks are predicted by a segmenter trained ONLY on the OTHER folds. Every
training image therefore receives a mask from a segmenter that never saw it -- exactly
the "unclean" regime it will face at test time. val/test masks come from the full-train
Stage-1 checkpoint (val/test are already held out of Stage-1 training).

The Stage-2 classifier is identical to train_shapeaware.py (EfficientNet-B0 + 6-shape
fusion); only the source of the training masks changes (OOF instead of in-fold), so the
gap vs. ImprovedDualStage_ShapeAware isolates the OOF correction.

Writes Models/results/ImprovedDualStage_ShapeAware_OOF.json.
Run:  python Models/ImprovedDualStage/train_shapeaware_oof.py
"""
import os, sys, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
PROJ  = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, PROJ)
sys.path.insert(0, _HERE)
from Models.common import data as D
from Models.common import metrics as M
from Models.common import runner as R
from Models.common.losses import ComboLoss
from Models.Improvement.model import build_model as build_improved_unet, predict_mask
from model import build_seg_model, roi_bbox_from_mask, crop_roi, CLS_NAMES, LABEL2ID
from train_shapeaware import (shape_features, ShapeAwareClassifier, ShapeROIDataset,
                              classify_tta, build_gt_items, build_pred_items)

NAME    = "ImprovedDualStage_ShapeAware_OOF"
K       = 5
SEG_EPOCHS = 40          # per-fold Stage-1 epochs (reduced; only need reasonable masks)
EPOCHS  = 60             # Stage-2 classifier epochs
BATCH   = 16
LR      = 1e-4
ROI_SZ  = 256
CKPT    = os.path.join(_HERE, "roi_classifier_shapeaware_oof.pt")


# --------------------------------------------------------------------------- #
# per-fold Stage-1 training (Improved-UNet recipe) + OOF mask prediction
# --------------------------------------------------------------------------- #
def train_segmenter(dff, epochs=SEG_EPOCHS):
    """Train an Improved-UNet on rows where dff.split=='train'."""
    device = R.DEVICE
    tr = D.SegDataset(dff, "train", augment=True, rgb3=True)
    loader = DataLoader(tr, batch_size=8, shuffle=True)
    model = build_improved_unet("efficientnet-b4", pretrained=True, arch="unet").to(device)
    crit = ComboLoss(alpha=0.3, beta=0.7, gamma=1.3333, bce_w=0.5, focal=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for ep in range(epochs):
        model.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); loss = crit(model(x), y); loss.backward(); opt.step()
        sched.step()
    return model


@torch.no_grad()
def predict_items(model, dff, split):
    """Build (crop, shape, label) items from PREDICTED masks for dff rows in `split`."""
    model.eval()
    items = []
    for _, row in dff[dff.split == split].iterrows():
        g, m = D.load_gray_and_mask(row)
        g256, m256, _ = D.letterbox(g, m)
        x = D._norm_to_tensor3(g256).unsqueeze(0).to(R.DEVICE)
        pr = predict_mask(model, x, thresh=0.5, tta=False, postprocess=True)[0]
        items.append((crop_roi(g256, roi_bbox_from_mask(pr), ROI_SZ),
                      shape_features(pr), LABEL2ID[row["cls"]]))
    return items


def build_oof_train_items(seg_df):
    """5-fold cross-fit over the TRAIN split: each fold's masks predicted by a
    segmenter trained on the other folds. Returns OOF (crop, shape, label) items."""
    tr = seg_df[seg_df.split == "train"].copy().reset_index(drop=True)
    clusters = tr["cluster"].unique()
    rng = np.random.RandomState(42); rng.shuffle(clusters)
    fold_of = {c: i % K for i, c in enumerate(clusters)}
    tr["fold"] = tr["cluster"].map(fold_of)
    oof = []
    for f in range(K):
        dff = tr.copy()
        dff["split"] = np.where(dff["fold"].values == f, "holdout", "train")
        t0 = time.time()
        model = train_segmenter(dff)
        items = predict_items(model, dff, "holdout")
        oof.extend(items)
        del model; torch.cuda.empty_cache()
        print(f"[{NAME}] fold {f+1}/{K}: trained on {int((dff.split=='train').sum())} "
              f"-> predicted {len(items)} holdout masks  ({(time.time()-t0)/60:.1f} min)")
    return oof


def main():
    R.set_seed(42)
    print(f"[{NAME}] device = {R.DEVICE}")
    df = D.make_split()

    # ---- OOF predicted masks for the TRAIN classifier inputs ----
    print(f"[{NAME}] building {K}-fold out-of-fold training masks (trains {K} segmenters)...")
    oof_tr = build_oof_train_items(df)

    # ---- val/test masks from the FULL-train Stage-1 (val/test already held out) ----
    seg_model, loaded = build_seg_model(load_checkpoint=True, device=R.DEVICE)
    print(f"[{NAME}] full Stage-1 checkpoint reused for val/test: {loaded}")
    pred_test, examples, gts, preds = build_pred_items(seg_model, df, "test", tta=True, want_examples=True)
    seg = M.aggregate_seg(preds, gts)
    gt_test = build_gt_items(df, "test")
    pred_va, *_ = build_pred_items(seg_model, df, "val")
    gt_tr = build_gt_items(df, "train")

    # Stage-2 training = GT items + OOF-predicted items (train/test match, no in-fold optimism)
    tr_items = gt_tr + oof_tr
    va_items = pred_va
    print(f"[{NAME}] Stage-2 train items {len(tr_items)} (gt {len(gt_tr)} + OOF-pred {len(oof_tr)})  val {len(va_items)}")

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

    results = {
        "model": "Shape-aware dual-stage, OUT-OF-FOLD training (Improved-UNet -> ROI+shape -> EfficientNet-B0*)",
        "paper": "Our extension of Bruno [17]: shape-feature fusion + K-fold cross-fitted training masks",
        "task": "segmentation-guided benign/malignant classification (2-class)",
        "our_work": "same shape-aware Stage-2 as ImprovedDualStage_ShapeAware, but the TRAINING masks "
                    f"are {K}-fold out-of-fold (each fold predicted by a segmenter trained on the other "
                    "folds), removing the in-fold optimism of the training masks (stacking / cross-fitting).",
        "config": {"img_size": D.IMG_SIZE, "roi_size": ROI_SZ, "k_folds": K, "seg_epochs_per_fold": SEG_EPOCHS,
                   "epochs": EPOCHS, "batch": BATCH, "lr": LR, "optimizer": "Adam", "scheduler": "cosine",
                   "stage1": "Improved Pretrained-UNet (OOF for train, full for val/test)",
                   "stage2": "EfficientNet-B0 + shape-feature fusion (6 descriptors)",
                   "train_data": "gt-mask + OUT-OF-FOLD predicted-mask items", "classes": list(CLS_NAMES)},
        "split": {k: int((df.split == k).sum()) for k in ["train", "val", "test"]},
        "best_val_macro_f1": round(best, 4), "train_minutes": round(train_min, 2),
        "test_segmentation": seg,
        "test_classification": cls_pipe,
        "test_classification_gt_roi": cls_gt,
        "history": hist,
        "baselines": {"bruno": 0.804, "improved_vanilla": 0.807, "strong_cls": 0.817,
                      "shape_aware_infold": 0.840},
    }
    R.save_results(NAME, results)
    print(f"[{NAME}] PIPELINE test  segDice={seg['dice']['mean']}  || cls macroF1={cls_pipe['macro_f1']}  "
          f"acc={cls_pipe['accuracy']}  AUC={cls_pipe['macro_auc_ovr']}  (GT-ROI F1={cls_gt['macro_f1']})")
    print(f"[{NAME}] vs: strong-cls 0.817 | shape-aware(in-fold) 0.840 | shape-aware(OOF) {cls_pipe['macro_f1']}")


if __name__ == "__main__":
    main()
