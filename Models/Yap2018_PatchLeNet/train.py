# -*- coding: utf-8 -*-
"""
Train + evaluate the Patch-based LeNet [02] on the cleaned/grouped BUSI split.

Yap et al. (2018) is a lesion **detection** method, so its native metrics are TPF /
FPs-per-image / F-measure (their eq. 11-13); Dice/IoU are reported only as a secondary
reference (a patch classifier is not a fine segmenter). To keep false positives under
control (as the paper does with small-region removal + ROI selection) we: mask the bright
top strip, then choose the decision threshold that maximises **validation** F-measure and
apply it to the test set.
Run:  python Models/Yap2018_PatchLeNet/train.py
Outputs: Models/results/Yap2018_PatchBasedLeNet.json (+ figures, checkpoint).
"""
import os, sys, time
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
PROJ  = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, PROJ)
from Models.common import data as D
from Models.common import metrics as M
from Models.common import runner as R
from Models.Yap2018_PatchLeNet.model import PatchLeNet

NAME     = "Yap2018_PatchBasedLeNet"
EPOCHS   = 40
BATCH    = 256
LR       = 0.01
STRIDE   = 12          # sliding-window stride at test time
MIN_AREA = 60          # drop tiny predicted blobs (Yap remove small FP regions)
TOP_CUT  = 0.05        # ignore the top 5% rows (bright skin/probe line -> false positives)
CKPT     = os.path.join(_HERE, "best.pt")
HALF     = D.PATCH // 2
# Yap et al. (2018) reported detection results (Table I) for reference/comparison:
PAPER = {"Patch-LeNet_A": {"TPF": 0.89, "FPs_per_image": 0.10, "F_measure": 0.88},
         "Patch-LeNet_B": {"TPF": 0.85, "FPs_per_image": 0.14, "F_measure": 0.86}}


@torch.no_grad()
def patch_accuracy(model, loader):
    model.eval(); correct = tot = 0
    for x, y in loader:
        pred = model(x.to(R.DEVICE)).argmax(1).cpu()
        correct += (pred == y).sum().item(); tot += len(y)
    return correct / max(tot, 1)


@torch.no_grad()
def compute_heat(model, gray):
    """Sliding window -> averaged lesion-probability map (before thresholding)."""
    centres = list(range(HALF, D.IMG_SIZE-HALF, STRIDE))
    patches = [gray[cy-HALF:cy+HALF, cx-HALF:cx+HALF] for cy in centres for cx in centres]
    coords  = [(cy, cx) for cy in centres for cx in centres]
    xb = torch.stack([D._norm_to_tensor(p) for p in patches]).to(R.DEVICE)
    probs = torch.softmax(model(xb), 1)[:, 1].cpu().numpy()
    accum = np.zeros((D.IMG_SIZE, D.IMG_SIZE), np.float32)
    count = np.zeros((D.IMG_SIZE, D.IMG_SIZE), np.float32)
    for (cy, cx), pr in zip(coords, probs):
        accum[cy-HALF:cy+HALF, cx-HALF:cx+HALF] += pr
        count[cy-HALF:cy+HALF, cx-HALF:cx+HALF] += 1
    return np.divide(accum, count, out=np.zeros_like(accum), where=count > 0)


def mask_from_heat(heat, valid, thresh, min_area=MIN_AREA, keep_top=2):
    """Threshold the heat map, drop tiny blobs, then keep only the `keep_top`
    highest-confidence regions (Yap's rule-based ROI selection) to limit false positives."""
    v = valid.copy(); v[:int(TOP_CUT*D.IMG_SIZE)] = 0          # kill top skin/probe strip
    mask = ((heat > thresh) & (v > 0)).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    regions = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            regions.append((float(heat[lab == i].mean()), i))   # rank by confidence
    regions.sort(reverse=True)
    out = np.zeros_like(mask)
    for _, i in regions[:keep_top]:
        out[lab == i] = 1
    return out


def main():
    R.set_seed(42)
    df = D.make_split()
    train_ds = D.PatchDataset(df, "train", per_image=80, augment=True)
    val_ds   = D.PatchDataset(df, "val",   per_image=40)
    va = DataLoader(val_ds, BATCH, shuffle=False)

    model = PatchLeNet().to(R.DEVICE)
    opt = torch.optim.RMSprop(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=12, gamma=0.3)
    crit = nn.CrossEntropyLoss()
    hist = {"train_loss": [], "val_patch_acc": []}
    best = -1
    print(f"[{NAME}] device={R.DEVICE}  train_patches={len(train_ds)} val_patches={len(val_ds)}")
    t0 = time.time()
    for ep in range(EPOCHS):
        train_ds.resample()
        tr = DataLoader(train_ds, BATCH, shuffle=True)
        model.train(); losses = []
        for x, y in tr:
            x, y = x.to(R.DEVICE), y.to(R.DEVICE)
            opt.zero_grad(); loss = crit(model(x), y); loss.backward(); opt.step()
            losses.append(loss.item())
        sched.step()
        acc = patch_accuracy(model, va)
        hist["train_loss"].append(float(np.mean(losses))); hist["val_patch_acc"].append(acc)
        if acc > best:
            best = acc; torch.save(model.state_dict(), CKPT)
        if ep % 5 == 0 or ep == EPOCHS-1:
            print(f"  ep{ep:02d}  loss {np.mean(losses):.4f}  val_patch_acc {acc:.4f}  (best {best:.4f})")
    train_min = (time.time()-t0)/60

    model.load_state_dict(torch.load(CKPT, map_location=R.DEVICE)); model.eval()

    # ---- choose decision threshold on VALIDATION (maximise F-measure) ----
    val_heat = [(compute_heat(model, g), m, v) for _, g, m, v in D.iter_test_images(df, "val")]
    grid = [0.4, 0.5, 0.6, 0.7, 0.8]
    best_thr, best_f = 0.5, -1
    for thr in grid:
        preds = [mask_from_heat(h, v, thr) for h, m, v in val_heat]
        gts   = [(m > 0).astype(np.uint8) for h, m, v in val_heat]
        f = M.detection_metrics(preds, gts, min_area=MIN_AREA)["F_measure"]
        print(f"  [val] thr={thr}  F={f:.4f}")
        if f > best_f:
            best_f, best_thr = f, thr
    print(f"[{NAME}] selected threshold={best_thr} (val F={best_f:.4f})")

    # ---- test with the selected threshold ----
    preds, gts, examples = [], [], []
    for row, g, m, v in D.iter_test_images(df, "test"):
        pr = mask_from_heat(compute_heat(model, g), v, best_thr)
        preds.append(pr); gts.append((m > 0).astype(np.uint8))
        if len(examples) < 6:
            examples.append((g, (m > 0).astype(np.uint8), pr))

    seg = M.aggregate_seg(preds, gts)
    det = M.detection_metrics(preds, gts, min_area=MIN_AREA)
    R.plot_curve(hist, NAME, keys=("train_loss", "val_patch_acc"))
    R.save_seg_examples(examples, NAME)

    results = {
        "model": "Patch-based LeNet", "paper": "Yap et al. (2018), IEEE JBHI [02]",
        "task": "lesion DETECTION via 28x28 patch classification + sliding window (benign+malignant)",
        "primary_metric": "detection (TPF / FPs_per_image / F_measure); Dice/IoU secondary",
        "our_work": "from-scratch PyTorch LeNet (+BatchNorm); balanced+hard-negative patch sampling; "
                    "sliding-window inference; val-tuned threshold; grouped leakage-free split",
        "config": {"patch": D.PATCH, "stride": STRIDE, "epochs": EPOCHS, "batch": BATCH,
                   "lr": LR, "optimizer": "RMSprop", "min_region_area": MIN_AREA,
                   "top_cut": TOP_CUT, "selected_threshold": best_thr,
                   "params_M": round(sum(p.numel() for p in model.parameters())/1e6, 3)},
        "split": {k: int((df.split == k).sum()) for k in ["train", "val", "test"]},
        "best_val_patch_acc": round(best, 4), "train_minutes": round(train_min, 2),
        "test_detection": det, "test_segmentation": seg,
        "paper_reported_detection": PAPER,
        "history": hist,
    }
    R.save_results(NAME, results)
    print(f"[{NAME}] TEST detection: TPF={det['TPF']}  FPs/img={det['FPs_per_image']}  F={det['F_measure']}  "
          f"| seg Dice={seg['dice']['mean']}  ({train_min:.1f} min)")


if __name__ == "__main__":
    main()
