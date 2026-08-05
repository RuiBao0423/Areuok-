# -*- coding: utf-8 -*-
"""
Models/PretrainedUNet_Improved/train.py

Trains the improved segmentation model and writes Models/results/<NAME>.json in
the SAME schema as the baselines (test_segmentation / test_detection / history /
config / split), so the notebook's comparison table picks it up automatically.

Run:
    python Models/PretrainedUNet_Improved/train.py
    python Models/PretrainedUNet_Improved/train.py --encoder resnet34 --arch unetpp
    python Models/PretrainedUNet_Improved/train.py --epochs 60 --loss dicebce   # ablation control

Everything else (cleaned grouped leakage-free split, letterbox preprocessing,
metrics, results/figure I/O) is REUSED from Models/common -- identical to the
baselines, so the numbers are directly comparable.
"""
import os, sys, time, json, argparse
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
from Models.Improvement.model import (
    build_model, count_params_m, predict_mask, predict_prob_tta)


# --------------------------------------------------------------------------- #
def evaluate(model, df, split, device, tta=True, postprocess=True, thresh=0.5):
    """Run the model over `split`, returning (seg_metrics, det_metrics, examples).
    Mirrors the baseline evaluation: per-image {0,1} masks -> common.metrics."""
    ds = D.SegDataset(df, split, augment=False, rgb3=True)
    loader = DataLoader(ds, batch_size=8, shuffle=False)
    model.eval()
    preds, gts, examples = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            masks = predict_mask(model, x, thresh=thresh, tta=tta, postprocess=postprocess)
            gt = (y.squeeze(1).cpu().numpy() > 0.5).astype(np.uint8)
            for i in range(len(masks)):
                preds.append(masks[i]); gts.append(gt[i])
                if len(examples) < 6:
                    # store grayscale (first channel, de-normalised approx) for the figure
                    g = x[i, 0].cpu().numpy()
                    examples.append((g, gt[i], masks[i]))
    seg = M.aggregate_seg(preds, gts)
    det = M.detection_metrics(preds, gts)
    return seg, det, examples


def val_dice(model, df, device):
    """Quick val Dice (no TTA/postproc) for model selection during training."""
    ds = D.SegDataset(df, "val", augment=False, rgb3=True)
    loader = DataLoader(ds, batch_size=8, shuffle=False)
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            prob = torch.sigmoid(model(x)).squeeze(1).cpu().numpy()
            for i in range(len(prob)):
                preds.append((prob[i] >= 0.5).astype(np.uint8))
                gts.append((y[i, 0].cpu().numpy() > 0.5).astype(np.uint8))
    return float(np.mean([M.dice(p, g) for p, g in zip(preds, gts)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="efficientnet-b4")
    ap.add_argument("--arch", default="unet", choices=["unet", "unetpp"])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--loss", default="focaltversky",
                    choices=["focaltversky", "tversky", "dicebce"])
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--no-post", action="store_true")
    ap.add_argument("--name", default=None)
    ap.add_argument("--epochs-smoke", type=int, default=0,
                    help="if >0, overrides epochs for a fast smoke test")
    args = ap.parse_args()
    if args.epochs_smoke:
        args.epochs = args.epochs_smoke

    R.set_seed(42)
    device = R.DEVICE
    name = args.name or f"PretrainedUNet_{args.encoder}_{args.arch}_{args.loss}"

    df = D.make_split(verbose=True)                       # cached leakage-free seg split
    tr = D.SegDataset(df, "train", augment=True,  rgb3=True)
    tr_loader = DataLoader(tr, batch_size=args.batch, shuffle=True, drop_last=False)

    model = build_model(args.encoder, pretrained=True, arch=args.arch).to(device)
    params_m = count_params_m(model)

    # loss selection (dicebce = symmetric control; focaltversky = the improvement)
    if args.loss == "focaltversky":
        crit = ComboLoss(alpha=0.3, beta=0.7, gamma=1.3333, bce_w=0.5, focal=True)
    elif args.loss == "tversky":
        crit = ComboLoss(alpha=0.3, beta=0.7, bce_w=0.5, focal=False)
    else:                                                 # dicebce control
        crit = ComboLoss(alpha=0.5, beta=0.5, bce_w=0.5, focal=False)
    crit = crit.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
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
            tot += float(loss); nb += 1
        sched.step()
        vd = val_dice(model, df, device)
        history["train_loss"].append(tot / max(nb, 1))
        history["val_dice"].append(vd)
        if vd > best_val:
            best_val = vd
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"[{name}] epoch {ep+1}/{args.epochs}  loss={tot/max(nb,1):.4f}  val_dice={vd:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    # persist the best Stage-1 segmenter so the dual-stage pipeline can reuse it
    ckpt_path = os.path.join(_HERE, "improved_unet_best.pt")
    torch.save({"state_dict": model.state_dict(),
                "encoder": args.encoder, "arch": args.arch}, ckpt_path)
    print(f"[{name}] saved checkpoint -> {ckpt_path}")
    train_minutes = round((time.time() - t0) / 60, 2)

    # ---- final test evaluation (TTA + post-processing = the full improvement) ----
    tta = not args.no_tta
    post = not args.no_post
    seg, det, examples = evaluate(model, df, "test", device, tta=tta, postprocess=post)

    result = {
        "model": f"Pretrained U-Net ({args.encoder}, {args.arch})",
        "paper": "Improvement on Ronneberger 2015 [03] / Oktay 2018 [09]",
        "task": "lesion segmentation (benign+malignant)",
        "our_work": ("ImageNet-pretrained {enc} encoder + U-Net decoder (smp); "
                     "Focal Tversky + BCE loss; TTA (hflip+multiscale) + "
                     "largest-CC post-processing; same grouped leakage-free split "
                     "and letterbox preprocessing as the baselines"
                     ).format(enc=args.encoder),
        "config": {
            "img_size": D.IMG_SIZE, "epochs": args.epochs, "batch": args.batch,
            "lr": args.lr, "optimizer": "AdamW", "scheduler": "cosine",
            "loss": args.loss, "tta": tta, "postprocess": post,
            "encoder": args.encoder, "arch": args.arch, "params_M": params_m,
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
          f"(baseline U-Net 0.648 / DeepLabV3+ 0.739)")
    return result


if __name__ == "__main__":
    main()
