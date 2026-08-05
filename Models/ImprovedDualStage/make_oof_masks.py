# -*- coding: utf-8 -*-
"""
Models/ImprovedDualStage/make_oof_masks.py

Generate and CACHE the predicted masks for the whole dual-stage classifier study, so
every downstream feature/ensemble experiment reuses them cheaply (seconds each) instead
of retraining segmenters.

  * TRAIN masks: 5-fold out-of-fold (cross-fitting) -- each fold's masks predicted by an
    Improved-UNet trained ONLY on the other folds (removes in-fold optimism).
  * VAL / TEST masks: predicted by the full-train Stage-1 checkpoint (val/test are already
    held out of Stage-1 training). Test uses TTA (matches final evaluation).

Cache -> Models/ImprovedDualStage/oof_cache.npz  with, per split, arrays:
    <split>_gray  (N,256,256) uint8   letterboxed grayscale
    <split>_gt    (N,256,256) uint8   ground-truth mask
    <split>_pred  (N,256,256) uint8   predicted mask (OOF for train, full for val/test)
    <split>_label (N,)        int64   0=benign, 1=malignant

Run once:  python Models/ImprovedDualStage/make_oof_masks.py   (~40 min, trains 5 segmenters)
"""
import os, sys, time
import numpy as np
import torch
from torch.utils.data import DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
PROJ  = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, PROJ)
sys.path.insert(0, _HERE)
from Models.common import data as D
from Models.common import runner as R
from Models.common.losses import ComboLoss
from Models.Improvement.model import build_model as build_improved_unet, predict_mask
from model import build_seg_model, LABEL2ID

K          = 5
SEG_EPOCHS = 40
CACHE      = os.path.join(_HERE, "oof_cache.npz")


def train_segmenter(dff, epochs=SEG_EPOCHS):
    """Train an Improved-UNet (EffB4 + Focal-Tversky) on dff rows where split=='train'."""
    device = R.DEVICE
    loader = DataLoader(D.SegDataset(dff, "train", augment=True, rgb3=True),
                        batch_size=8, shuffle=True)
    model = build_improved_unet("efficientnet-b4", pretrained=True, arch="unet").to(device)
    crit = ComboLoss(alpha=0.3, beta=0.7, gamma=1.3333, bce_w=0.5, focal=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for _ in range(epochs):
        model.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); crit(model(x), y).backward(); opt.step()
        sched.step()
    return model


@torch.no_grad()
def predict_rows(model, rows, tta):
    """Return gray/gt/pred/label arrays for the given dataframe rows."""
    model.eval()
    gray, gt, pred, label = [], [], [], []
    for _, row in rows.iterrows():
        g, m = D.load_gray_and_mask(row)
        g256, m256, _ = D.letterbox(g, m)
        x = D._norm_to_tensor3(g256).unsqueeze(0).to(R.DEVICE)
        pr = predict_mask(model, x, thresh=0.5, tta=tta, postprocess=True)[0]
        gray.append(g256); gt.append((m256 > 0).astype(np.uint8))
        pred.append(pr); label.append(LABEL2ID[row["cls"]])
    return (np.stack(gray).astype(np.uint8), np.stack(gt).astype(np.uint8),
            np.stack(pred).astype(np.uint8), np.array(label, np.int64))


def main():
    R.set_seed(42)
    print(f"[make_oof] device = {R.DEVICE}")
    df = D.make_split()

    # ---------- TRAIN: 5-fold out-of-fold masks ----------
    tr = df[df.split == "train"].copy().reset_index(drop=True)
    clusters = tr["cluster"].unique()
    rng = np.random.RandomState(42); rng.shuffle(clusters)
    fold_of = {c: i % K for i, c in enumerate(clusters)}
    tr["fold"] = tr["cluster"].map(fold_of)

    N = len(tr)
    tr_gray = np.zeros((N, 256, 256), np.uint8)
    tr_gt   = np.zeros((N, 256, 256), np.uint8)
    tr_pred = np.zeros((N, 256, 256), np.uint8)
    tr_lab  = np.zeros(N, np.int64)
    t_all = time.time()
    for f in range(K):
        dff = tr.copy()
        dff["split"] = np.where(dff["fold"].values == f, "holdout", "train")
        t0 = time.time()
        model = train_segmenter(dff)
        hold = dff[dff["split"] == "holdout"]
        g, gt, pr, lab = predict_rows(model, hold, tta=False)
        pos = hold.index.to_numpy()
        tr_gray[pos] = g; tr_gt[pos] = gt; tr_pred[pos] = pr; tr_lab[pos] = lab
        del model; torch.cuda.empty_cache()
        print(f"[make_oof] fold {f+1}/{K}: train {int((dff.split=='train').sum())} "
              f"-> {len(hold)} OOF masks  ({(time.time()-t0)/60:.1f} min)")

    # ---------- VAL / TEST: full-train Stage-1 ----------
    seg_model, loaded = build_seg_model(load_checkpoint=True, device=R.DEVICE)
    print(f"[make_oof] full Stage-1 reused for val/test: {loaded}")
    va = predict_rows(seg_model, df[df.split == "val"], tta=False)
    te = predict_rows(seg_model, df[df.split == "test"], tta=True)

    np.savez_compressed(
        CACHE,
        tr_gray=tr_gray, tr_gt=tr_gt, tr_pred=tr_pred, tr_label=tr_lab,
        va_gray=va[0], va_gt=va[1], va_pred=va[2], va_label=va[3],
        te_gray=te[0], te_gt=te[1], te_pred=te[2], te_label=te[3],
    )
    print(f"[make_oof] wrote {CACHE}  (train {N}, val {len(va[3])}, test {len(te[3])})  "
          f"total {(time.time()-t_all)/60:.1f} min")


if __name__ == "__main__":
    main()
