# -*- coding: utf-8 -*-
"""
9444_07_deeplab/train.py
Train + evaluate DeepLabV3+ [paper 7] on the BUSI lesion-segmentation split.

Reuses the shared pipeline in common/:
  * data.make_split()        -> leakage-free train/val/test (benign+malignant)
  * data.SegDataset(rgb3=True) -> 3-channel ImageNet-normalised images + masks
  * metrics.aggregate_seg()  -> Dice / IoU / pixel acc-prec-rec on the test set
  * runner.*                 -> seeding, DEVICE, results JSON, curves, example grid

Loss is BCE + soft-Dice (standard for imbalanced medical segmentation). Selection
is by best validation Dice. Run:  python train.py
"""
import os, sys, time
_HERE = os.path.dirname(os.path.abspath(__file__))
PROJ  = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
# _HERE holds model.py; PROJ makes the shared `Models.common` package importable
sys.path.insert(0, PROJ)
sys.path.insert(0, _HERE)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from Models.common import data, metrics, runner
from model import DeepLab

# --------------------------- hyper-parameters --------------------------- #
EPOCHS     = 60
BATCH      = 8
LR         = 1e-4
WEIGHT_DECAY = 1e-4
NAME       = "DeepLabV3plus_Chen2018"            # -> Models/results/DeepLabV3plus_Chen2018.json + figures
NUM_WORKERS = 0


# --------------------------- losses / helpers --------------------------- #
def soft_dice_loss(logits, target, eps=1e-6):
    """1 - soft Dice over the (single) foreground channel; differentiable."""
    prob = torch.sigmoid(logits)
    p = prob.flatten(1)
    t = target.flatten(1)
    inter = (p * t).sum(1)
    dice = (2 * inter + eps) / (p.sum(1) + t.sum(1) + eps)
    return 1 - dice.mean()


@torch.no_grad()
def evaluate(model, loader, collect_examples=False):
    """Return mean Dice over `loader`; optionally also (preds, gts, examples)
    for the final test report."""
    model.eval()
    dices, preds, gts, examples = [], [], [], []
    for x, y in loader:
        x = x.to(runner.DEVICE)
        pred = (torch.sigmoid(model(x)) > 0.5).cpu().numpy().astype(np.uint8)
        yn = y.numpy().astype(np.uint8)
        xn = x.cpu().numpy()
        for pi, yi, xi in zip(pred, yn, xn):
            dices.append(metrics.dice(pi[0], yi[0]))
            if collect_examples:
                preds.append(pi[0]); gts.append(yi[0])
                # display channel: undo ImageNet norm roughly -> a viewable gray
                g = xi[0] * 0.229 + 0.485
                examples.append((g, yi[0], pi[0]))
    mean_dice = float(np.mean(dices)) if dices else 0.0
    if collect_examples:
        return mean_dice, preds, gts, examples
    return mean_dice


def main():
    runner.set_seed(42)
    print(f"[train] device = {runner.DEVICE}")

    df = data.make_split()                       # shared, cached, leakage-free split
    tr_ds = data.SegDataset(df, "train", augment=True,  rgb3=True)
    va_ds = data.SegDataset(df, "val",   augment=False, rgb3=True)
    te_ds = data.SegDataset(df, "test",  augment=False, rgb3=True)
    print(f"[train] sizes -> train {len(tr_ds)} | val {len(va_ds)} | test {len(te_ds)}")

    tr = DataLoader(tr_ds, batch_size=BATCH, shuffle=True,
                    num_workers=NUM_WORKERS, drop_last=True)
    va = DataLoader(va_ds, batch_size=BATCH, shuffle=False, num_workers=NUM_WORKERS)
    te = DataLoader(te_ds, batch_size=BATCH, shuffle=False, num_workers=NUM_WORKERS)

    model = DeepLab(num_classes=1, output_stride=16, pretrained=True).to(runner.DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    bce = nn.BCEWithLogitsLoss()

    hist = {"train_loss": [], "val_dice": []}
    best_dice, best_state = -1.0, None

    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running = 0.0
        for x, y in tr:
            x, y = x.to(runner.DEVICE), y.to(runner.DEVICE)
            opt.zero_grad()
            logits = model(x)
            loss = bce(logits, y) + soft_dice_loss(logits, y)
            loss.backward()
            opt.step()
            running += loss.item()
        sched.step()

        train_loss = running / max(len(tr), 1)
        val_dice = evaluate(model, va)
        hist["train_loss"].append(train_loss)
        hist["val_dice"].append(val_dice)
        print(f"[epoch {epoch:3d}/{EPOCHS}] train_loss={train_loss:.4f}  val_dice={val_dice:.4f}")

        if val_dice > best_dice:
            best_dice = val_dice
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    train_min = (time.time() - t0) / 60

    # restore best-on-val weights before the final test evaluation
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"[train] best val_dice = {best_dice:.4f}; evaluating on test...")

    _, preds, gts, examples = evaluate(model, te, collect_examples=True)
    seg = metrics.aggregate_seg(preds, gts)
    det = metrics.detection_metrics(preds, gts)

    # save weights + results + figures next to the other baselines
    torch.save(model.state_dict(), os.path.join(_HERE, f"{NAME}.pt"))
    runner.plot_curve(hist, NAME, keys=("train_loss", "val_dice"))
    runner.save_seg_examples(examples, NAME, n=6)

    results = {
        "model": "DeepLabV3+ (ResNet-50, OS=16)",
        "paper": "Chen et al. (2018), DeepLabV3+ [07]",
        "task": "lesion segmentation (benign+malignant)",
        "our_work": "wrote the ASPP + atrous-separable decoder head (paper Fig. 2); "
                    "ImageNet-pretrained ResNet-50 backbone with dilated last stage; "
                    "BCE+soft-Dice loss; shared letterbox preprocessing + grouped "
                    "leakage-free split (see README.md)",
        "config": {"img_size": data.IMG_SIZE, "epochs": EPOCHS, "batch": BATCH, "lr": LR,
                   "optimizer": "AdamW(wd=1e-4)+CosineAnnealing", "loss": "BCE+soft-Dice",
                   "output_stride": 16,
                   "params_M": round(sum(p.numel() for p in model.parameters())/1e6, 2)},
        "split": {k: int((df.split == k).sum()) for k in ["train", "val", "test"]},
        "best_val_dice": round(best_dice, 4), "train_minutes": round(train_min, 2),
        "test_segmentation": seg, "test_detection": det,
        "history": hist,
    }
    runner.save_results(NAME, results)
    print(f"[{NAME}] TEST  Dice={seg['dice']['mean']}  IoU={seg['iou']['mean']}  "
          f"F-measure={det['F_measure']}  ({train_min:.1f} min)")


if __name__ == "__main__":
    main()
