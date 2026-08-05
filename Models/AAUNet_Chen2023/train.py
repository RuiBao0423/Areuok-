# -*- coding: utf-8 -*-
"""
Models/AAUNet_Chen2023/train.py

Trains AAU-Net [12] and writes Models/results/<NAME>.json in the SAME schema as the
segmentation baselines, so the notebook comparison table picks it up automatically.
Uses the shared cleaned, grouped, leakage-free seg split + letterbox preprocessing ->
directly comparable to U-Net / DeepLabV3+ / nnU-Net / the Improved model.

Paper: Adam lr 1e-3, 50 epochs, batch 12, BCE loss, 4-fold CV. Faithful deviations
(README): we use the project's single leakage-free split (not 4-fold) and Dice+BCE
(the repo convention; paper is BCE-only) for a fair, stable comparison.

Run:
    python Models/AAUNet_Chen2023/train.py
    python Models/AAUNet_Chen2023/train.py --epochs 80
"""
import os, sys, time, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
PROJ  = os.path.dirname(os.path.dirname(_HERE))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

from Models.common import data as D
from Models.common import metrics as M
from Models.common import runner as R
from Models.common.losses import ComboLoss
from Models.AAUNet_Chen2023.model import AAUNet, count_params_m


def val_dice(model, df, device):
    ds = D.SegDataset(df, "val", augment=False, rgb3=False)
    loader = DataLoader(ds, batch_size=8, shuffle=False)
    model.eval(); scores = []
    with torch.no_grad():
        for x, y in loader:
            prob = torch.sigmoid(model(x.to(device))).squeeze(1).cpu().numpy()
            for i in range(len(prob)):
                scores.append(M.dice((prob[i] >= 0.5).astype(np.uint8),
                                     (y[i, 0].numpy() > 0.5).astype(np.uint8)))
    return float(np.mean(scores))


def evaluate(model, df, device):
    ds = D.SegDataset(df, "test", augment=False, rgb3=False)
    loader = DataLoader(ds, batch_size=8, shuffle=False)
    model.eval(); preds, gts, examples = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            prob = torch.sigmoid(model(x)).squeeze(1).cpu().numpy()
            gt = (y.squeeze(1).numpy() > 0.5).astype(np.uint8)
            for i in range(len(prob)):
                pr = (prob[i] >= 0.5).astype(np.uint8)
                preds.append(pr); gts.append(gt[i])
                if len(examples) < 6:
                    examples.append((x[i, 0].cpu().numpy(), gt[i], pr))
    return M.aggregate_seg(preds, gts), M.detection_metrics(preds, gts), examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--name", default="AAUNet_Chen2023")
    ap.add_argument("--epochs-smoke", type=int, default=0)
    args = ap.parse_args()
    if args.epochs_smoke:
        args.epochs = args.epochs_smoke

    R.set_seed(42)
    device = R.DEVICE
    name = args.name

    df = D.make_split(verbose=True)
    tr = D.SegDataset(df, "train", augment=True, rgb3=False)
    tr_loader = DataLoader(tr, batch_size=args.batch, shuffle=True)

    model = AAUNet(in_ch=1, n_classes=1).to(device)
    params_m = count_params_m(model)
    crit = ComboLoss(alpha=0.5, beta=0.5, bce_w=0.5, focal=False).to(device)  # Dice+BCE
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    history = {"train_loss": [], "val_dice": []}
    best_val, best_state = -1.0, None
    t0 = time.time()
    for ep in range(args.epochs):
        model.train(); tot = 0.0; nb = 0
        for x, y in tr_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward(); opt.step()
            tot += float(loss.detach()); nb += 1
        sched.step()
        vd = val_dice(model, df, device)
        history["train_loss"].append(tot / max(nb, 1))
        history["val_dice"].append(vd)
        if vd > best_val:
            best_val = vd
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"[{name}] epoch {ep+1}/{args.epochs}  loss={tot/max(nb,1):.4f}  val_dice={vd:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    train_minutes = round((time.time() - t0) / 60, 2)

    seg, det, examples = evaluate(model, df, device)
    result = {
        "model": "AAU-Net (Hybrid Adaptive Attention U-Net)",
        "paper": "Chen et al. (2023), AAU-Net, IEEE TMI [12]",
        "task": "lesion segmentation (benign+malignant)",
        "our_work": ("U-Net with the HAAM (hybrid adaptive attention module) written "
                     "from scratch: parallel 3x3/5x5/dilated-3 convs + channel "
                     "self-attention (alpha/1-alpha branch routing) + spatial "
                     "self-attention (beta/1-beta); two HAAMs per stage. Same grouped "
                     "leakage-free split + letterbox as the baselines (comparable)."),
        "config": {
            "img_size": D.IMG_SIZE, "epochs": args.epochs, "batch": args.batch,
            "lr": args.lr, "optimizer": "Adam", "scheduler": "cosine",
            "loss": "Dice+BCE (paper: BCE only)", "features": [32, 64, 128, 256, 512],
            "params_M": params_m,
        },
        "split": {k: int((df.split == k).sum()) for k in ("train", "val", "test")},
        "best_val_dice": round(best_val, 4),
        "train_minutes": train_minutes,
        "test_segmentation": seg,
        "test_detection": det,
        "history": history,
        "paper_reported_busi": {"dice": 0.7751, "iou": 0.6882},
    }
    R.save_results(name, result)
    R.plot_curve(history, name, keys=("train_loss", "val_dice"))
    R.save_seg_examples(examples, name, n=min(6, len(examples)))
    print(f"[{name}]  test Dice={seg['dice']['mean']:.4f}  IoU={seg['iou']['mean']:.4f}  "
          f"(paper BUSI 0.775 | our U-Net 0.648 / DeepLabV3+ 0.739 / Improved 0.759)")
    return result


if __name__ == "__main__":
    main()
