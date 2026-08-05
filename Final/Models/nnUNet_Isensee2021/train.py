# -*- coding: utf-8 -*-
"""
Models/nnUNet_Isensee2021/train.py

Trains the nnU-Net-blueprint 2D U-Net and writes Models/results/<NAME>.json in the
SAME schema as the segmentation baselines (test_segmentation / test_detection /
history / config / split), so the notebook comparison table picks it up automatically.

Faithful nnU-Net training recipe (blueprint parameters):
  * loss   = Cross-Entropy + soft Dice, deep-supervised across resolutions with
             weights [1, 1/2, 1/4, ...] normalised to sum to 1  (binary lesion ->
             BCE is the 2-class Cross-Entropy)
  * optim  = SGD, Nesterov momentum 0.99, weight decay 3e-5
  * lr     = 0.01 with polyLR schedule  lr = lr0 * (1 - t/T)^0.9

Deviations from full nnU-Net (documented in README.md): epochs scaled down from
1000x250 iters to a fixed budget; shared repo preprocessing (letterbox + [-1,1]) and
augmentation are reused from Models/common instead of nnU-Net's z-score +
batchgenerators pipeline, so the split/preprocessing are IDENTICAL to the other
segmentation baselines and the Dice numbers are directly comparable.

Run:
    python Models/nnUNet_Isensee2021/train.py
    python Models/nnUNet_Isensee2021/train.py --epochs 200
"""
import os, sys, time, argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
PROJ  = os.path.dirname(os.path.dirname(_HERE))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

from Models.common import data as D
from Models.common import metrics as M
from Models.common import runner as R
from Models.nnUNet_Isensee2021.model import NNUNet, count_params_m


# --------------------------------------------------------------------------- #
# deep-supervised Cross-Entropy(=BCE for binary) + soft Dice loss
# --------------------------------------------------------------------------- #
def _soft_dice(logits, target, eps=1.0):
    p = torch.sigmoid(logits)
    p = p.reshape(p.shape[0], -1)
    g = target.reshape(target.shape[0], -1)
    inter = (p * g).sum(1)
    return (1 - (2 * inter + eps) / (p.sum(1) + g.sum(1) + eps)).mean()


def deep_supervision_loss(outputs, target):
    """outputs: list of logits [full-res, half, quarter, ...]. target: (N,1,256,256).
    nnU-Net weights halve at each lower resolution, normalised to sum to 1."""
    weights = [0.5 ** k for k in range(len(outputs))]
    s = sum(weights)
    weights = [w / s for w in weights]
    loss = 0.0
    for w, out in zip(weights, outputs):
        if out.shape[-2:] != target.shape[-2:]:
            gt = F.interpolate(target, size=out.shape[-2:], mode="nearest")
        else:
            gt = target
        bce = F.binary_cross_entropy_with_logits(out, gt)
        loss = loss + w * (bce + _soft_dice(out, gt))
    return loss


# --------------------------------------------------------------------------- #
def val_dice(model, df, device):
    ds = D.SegDataset(df, "val", augment=False, rgb3=False)
    loader = DataLoader(ds, batch_size=8, shuffle=False)
    model.eval()
    scores = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            prob = torch.sigmoid(model(x, deep=False)).squeeze(1).cpu().numpy()
            for i in range(len(prob)):
                pr = (prob[i] >= 0.5).astype(np.uint8)
                gt = (y[i, 0].numpy() > 0.5).astype(np.uint8)
                scores.append(M.dice(pr, gt))
    return float(np.mean(scores))


def evaluate(model, df, device):
    ds = D.SegDataset(df, "test", augment=False, rgb3=False)
    loader = DataLoader(ds, batch_size=8, shuffle=False)
    model.eval()
    preds, gts, examples = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            prob = torch.sigmoid(model(x, deep=False)).squeeze(1).cpu().numpy()
            gt = (y.squeeze(1).numpy() > 0.5).astype(np.uint8)
            for i in range(len(prob)):
                pr = (prob[i] >= 0.5).astype(np.uint8)
                preds.append(pr); gts.append(gt[i])
                if len(examples) < 6:
                    examples.append((x[i, 0].cpu().numpy(), gt[i], pr))
    return M.aggregate_seg(preds, gts), M.detection_metrics(preds, gts), examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--name", default="nnUNet_Isensee2021")
    ap.add_argument("--epochs-smoke", type=int, default=0)
    args = ap.parse_args()
    if args.epochs_smoke:
        args.epochs = args.epochs_smoke

    R.set_seed(42)
    device = R.DEVICE
    name = args.name

    df = D.make_split(verbose=True)                        # cached leakage-free seg split
    tr = D.SegDataset(df, "train", augment=True, rgb3=False)
    tr_loader = DataLoader(tr, batch_size=args.batch, shuffle=True, drop_last=False)

    model = NNUNet(in_ch=1, n_classes=1, features=(32, 64, 128, 256, 512), n_ds=3).to(device)
    params_m = count_params_m(model)

    # nnU-Net optimiser: SGD, Nesterov momentum 0.99, weight decay 3e-5
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.99,
                          nesterov=True, weight_decay=3e-5)
    # polyLR: lr = lr0 * (1 - t/T)^0.9
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda=lambda ep: (1 - ep / args.epochs) ** 0.9)

    history = {"train_loss": [], "val_dice": []}
    best_val, best_state = -1.0, None
    t0 = time.time()
    for ep in range(args.epochs):
        model.train(); tot = 0.0; nb = 0
        for x, y in tr_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            outs = model(x, deep=True)
            loss = deep_supervision_loss(outs, y)
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
        "model": "nnU-Net (2D blueprint)",
        "paper": "Isensee et al. (2021), nnU-Net, Nature Methods [20]",
        "task": "lesion segmentation (benign+malignant)",
        "our_work": ("2D U-Net blueprint written from scratch: InstanceNorm + "
                     "LeakyReLU(0.01), strided-conv downsampling, transposed-conv "
                     "upsampling, deep supervision; CE+Dice deep-supervised loss; "
                     "SGD Nesterov 0.99 + polyLR^0.9. Same grouped leakage-free split "
                     "and letterbox preprocessing as the baselines (directly comparable)."),
        "config": {
            "img_size": D.IMG_SIZE, "epochs": args.epochs, "batch": args.batch,
            "lr": args.lr, "optimizer": "SGD(nesterov,0.99)", "scheduler": "polyLR^0.9",
            "loss": "CE+Dice (deep-supervised)", "features": [32, 64, 128, 256, 512],
            "deep_supervision_heads": 3, "params_M": params_m,
        },
        "split": {k: int((df.split == k).sum()) for k in ("train", "val", "test")},
        "best_val_dice": round(best_val, 4),
        "train_minutes": train_minutes,
        "test_segmentation": seg,
        "test_detection": det,
        "history": history,
    }
    R.save_results(name, result)
    R.plot_curve(history, name, keys=("train_loss", "val_dice"))
    R.save_seg_examples(examples, name, n=min(6, len(examples)))
    print(f"[{name}]  test Dice={seg['dice']['mean']:.4f}  IoU={seg['iou']['mean']:.4f}  "
          f"(baseline U-Net 0.648 / DeepLabV3+ 0.739 / Improved 0.759)")
    return result


if __name__ == "__main__":
    main()
