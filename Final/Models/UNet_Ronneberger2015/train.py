# -*- coding: utf-8 -*-
"""
Train + evaluate U-Net [03] on the cleaned/grouped BUSI split and write metrics JSON.
Run:  python Models/UNet_Ronneberger2015/train.py
Outputs: Models/results/UNet_Ronneberger2015.json (+ figures, checkpoint).
"""
import os, sys, time, json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
PROJ  = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, PROJ)
from Models.common import data as D
from Models.common import metrics as M
from Models.common import runner as R
from Models.UNet_Ronneberger2015.model import UNet

NAME   = "UNet_Ronneberger2015"
EPOCHS = 60
BATCH  = 8
LR     = 1e-3
CKPT   = os.path.join(_HERE, "best.pt")


def dice_bce_loss(logits, target):
    """Dice + BCE  (chosen because EDA 3.5 shows many tiny lesions -> pixel imbalance)."""
    bce = nn.functional.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    inter = (p*target).sum((1, 2, 3))
    dl = 1 - (2*inter + 1) / (p.sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1)
    return bce + dl.mean()


@torch.no_grad()
def evaluate(model, loader):
    model.eval(); ds = []
    for x, y in loader:
        x = x.to(R.DEVICE)
        p = torch.sigmoid(model(x)).cpu().numpy() > 0.5
        for pi, yi in zip(p, y.numpy()):
            ds.append(M.dice(pi[0], yi[0] > 0.5))
    return float(np.mean(ds))


def main():
    R.set_seed(42)
    df = D.make_split()
    tr = DataLoader(D.SegDataset(df, "train", augment=True), BATCH, shuffle=True, num_workers=0)
    va = DataLoader(D.SegDataset(df, "val"),   BATCH, shuffle=False, num_workers=0)

    model = UNet(1, 1, base=64).to(R.DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    hist = {"train_loss": [], "val_dice": []}
    best = -1
    print(f"[{NAME}] device={R.DEVICE}  train={len(tr.dataset)} val={len(va.dataset)}")
    t0 = time.time()
    for ep in range(EPOCHS):
        model.train(); losses = []
        for x, y in tr:
            x, y = x.to(R.DEVICE), y.to(R.DEVICE)
            opt.zero_grad()
            loss = dice_bce_loss(model(x), y)
            loss.backward(); opt.step()
            losses.append(loss.item())
        sched.step()
        vd = evaluate(model, va)
        hist["train_loss"].append(float(np.mean(losses))); hist["val_dice"].append(vd)
        if vd > best:
            best = vd; torch.save(model.state_dict(), CKPT)
        if ep % 5 == 0 or ep == EPOCHS-1:
            print(f"  ep{ep:02d}  loss {np.mean(losses):.4f}  val_dice {vd:.4f}  (best {best:.4f})")
    train_min = (time.time()-t0)/60

    # ---- test with best checkpoint ----
    model.load_state_dict(torch.load(CKPT, map_location=R.DEVICE)); model.eval()
    preds, gts, examples = [], [], []
    with torch.no_grad():
        for row, g, m, v in D.iter_test_images(df, "test"):
            x = D._norm_to_tensor(g).unsqueeze(0).to(R.DEVICE)
            pr = (torch.sigmoid(model(x)).cpu().numpy()[0, 0] > 0.5).astype(np.uint8)
            preds.append(pr); gts.append((m > 0).astype(np.uint8))
            if len(examples) < 6:
                examples.append((g, (m > 0).astype(np.uint8), pr))

    seg = M.aggregate_seg(preds, gts)
    det = M.detection_metrics(preds, gts)
    R.plot_curve(hist, NAME, keys=("train_loss", "val_dice"))
    R.save_seg_examples(examples, NAME)

    results = {
        "model": "U-Net", "paper": "Ronneberger et al. (2015), MICCAI [03]",
        "task": "lesion segmentation (benign+malignant)",
        "our_work": "from-scratch PyTorch U-Net; Dice+BCE loss; letterbox preprocessing; "
                    "grouped leakage-free split (all our own code, see README.md)",
        "config": {"img_size": D.IMG_SIZE, "epochs": EPOCHS, "batch": BATCH, "lr": LR,
                   "optimizer": "Adam", "loss": "Dice+BCE", "params_M": round(
                       sum(p.numel() for p in model.parameters())/1e6, 2)},
        "split": {k: int((df.split == k).sum()) for k in ["train", "val", "test"]},
        "best_val_dice": round(best, 4), "train_minutes": round(train_min, 2),
        "test_segmentation": seg, "test_detection": det,
        "history": hist,
    }
    R.save_results(NAME, results)
    print(f"[{NAME}] TEST  Dice={seg['dice']['mean']}  IoU={seg['iou']['mean']}  "
          f"F-measure={det['F_measure']}  ({train_min:.1f} min)")


if __name__ == "__main__":
    main()
