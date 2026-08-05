# -*- coding: utf-8 -*-
"""
Models/MTL_OCA_Lu2025/train.py

Trains the MTL-OCA multi-task model (Lu et al. 2025 [25]) and writes
Models/results/<NAME>.json holding BOTH a segmentation block (test_segmentation /
test_detection) AND a classification block (test_classification), so the notebook can
compare it against the segmentation baselines AND the classification baselines.

Combined loss:  L = 0.4 * CE(soft_mask) + CE(final_mask) + CE(cls)   (alpha=0.4)
Optimizer: Adam, lr 1e-3 (paper). Uses the SAME cleaned, grouped, leakage-free
3-class split (make_cls_split) as DenseNet/EfficientNet, so the classification number
is directly comparable; segmentation is scored on the benign+malignant subset of that
same test set (leakage-free, but a different image set than the seg-only 98-img split).

Run:
    python Models/MTL_OCA_Lu2025/train.py
    python Models/MTL_OCA_Lu2025/train.py --epochs 150
"""
import os, sys, time, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
PROJ  = os.path.dirname(os.path.dirname(_HERE))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

from Models.common import data as D
from Models.common import metrics as M
from Models.common import runner as R
from Models.MTL_OCA_Lu2025.model import MTLOCA, MTLDataset, count_params_m

ALPHA = 0.4                                              # intra-segmentation weight (paper)


def mtl_loss(coarse, final, logits, mask, label, cls_w):
    l_soft = F.cross_entropy(coarse, mask)
    l_aug  = F.cross_entropy(final, mask)
    l_cls  = F.cross_entropy(logits, label, weight=cls_w)
    return ALPHA * l_soft + l_aug + l_cls


@torch.no_grad()
def run_split(model, loader, device, is_lesion):
    """Return (seg_preds, seg_gts on benign+malignant), (y_true,y_pred,y_proba all 3
    classes). is_lesion: bool array aligned to the split rows (cls in benign/malignant).
    `loader` is prebuilt once and reused every epoch (no per-epoch disk reload)."""
    model.eval()
    seg_preds, seg_gts = [], []
    y_true, y_pred, y_proba = [], [], []
    idx = 0
    for x, mask, label in loader:
        x = x.to(device)
        coarse, final, logits = model(x)
        pred = final.argmax(1).cpu().numpy().astype(np.uint8)         # (B,H,W) fg mask
        gt = mask.numpy().astype(np.uint8)
        proba = torch.softmax(logits, 1).cpu().numpy()
        for b in range(len(pred)):
            if is_lesion[idx]:                            # seg scored on lesion images only
                seg_preds.append(pred[b]); seg_gts.append(gt[b])
            y_true.append(int(label[b])); y_pred.append(int(proba[b].argmax()))
            y_proba.append(proba[b].tolist())
            idx += 1
    return seg_preds, seg_gts, (y_true, y_pred, y_proba)


def lesion_flags(df, split):
    rows = df[df.split == split].reset_index(drop=True)
    return (rows["cls"].isin(["benign", "malignant"]).values)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--name", default="MTL_OCA_Lu2025")
    ap.add_argument("--epochs-smoke", type=int, default=0)
    args = ap.parse_args()
    if args.epochs_smoke:
        args.epochs = args.epochs_smoke

    R.set_seed(42)
    device = R.DEVICE
    name = args.name

    df = D.make_cls_split(verbose=True)                   # cached leakage-free 3-class split
    tr = MTLDataset(df, "train", augment=True)
    tr_loader = DataLoader(tr, batch_size=args.batch, shuffle=True, drop_last=False)
    # build val/test loaders ONCE and reuse every epoch (no per-epoch disk reload)
    val_loader  = DataLoader(MTLDataset(df, "val",  augment=False), batch_size=8, shuffle=False)
    test_loader = DataLoader(MTLDataset(df, "test", augment=False), batch_size=8, shuffle=False)
    cls_w = D.class_weights(df).to(device)
    val_lesion = lesion_flags(df, "val")
    test_lesion = lesion_flags(df, "test")

    model = MTLOCA(in_ch=2, n_cls=len(D.CLASS_NAMES), K=2).to(device)
    params_m = count_params_m(model)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    history = {"train_loss": [], "val_dice": [], "val_macro_f1": []}
    best_score, best_state = -1.0, None
    t0 = time.time()
    for ep in range(args.epochs):
        model.train(); tot = 0.0; nb = 0
        for x, mask, label in tr_loader:
            x, mask, label = x.to(device), mask.to(device), label.to(device)
            opt.zero_grad()
            coarse, final, logits = model(x)
            loss = mtl_loss(coarse, final, logits, mask, label, cls_w)
            loss.backward(); opt.step()
            tot += float(loss.detach()); nb += 1
        sched.step()

        sp, sg, (yt, yp, ypr) = run_split(model, val_loader, device, val_lesion)
        vd = float(np.mean([M.dice(p, g) for p, g in zip(sp, sg)]))
        vf1 = M.classification_metrics(yt, yp, ypr, list(D.CLASS_NAMES))["macro_f1"]
        history["train_loss"].append(tot / max(nb, 1))
        history["val_dice"].append(vd); history["val_macro_f1"].append(vf1)
        score = vd + vf1                                  # joint model-selection score
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"[{name}] epoch {ep+1}/{args.epochs}  loss={tot/max(nb,1):.4f}  "
                  f"val_dice={vd:.4f}  val_macroF1={vf1:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    train_minutes = round((time.time() - t0) / 60, 2)

    # ---- final test evaluation: seg (benign+malignant) + cls (all 3 classes) ----
    sp, sg, (yt, yp, ypr) = run_split(model, test_loader, device, test_lesion)
    seg = M.aggregate_seg(sp, sg)
    det = M.detection_metrics(sp, sg)
    cls = M.classification_metrics(yt, yp, ypr, list(D.CLASS_NAMES))

    result = {
        "model": "MTL-OCA (Res-UNet + Object Contextual Attention)",
        "paper": "Lu et al. (2025), Front. Oncol. 15:1567577 [25]",
        "task": "multi-task: lesion segmentation + 3-class classification",
        "our_work": ("shared Res-UNet backbone (residual + GroupNorm) written from "
                     "scratch; OCA (OCR-style object-contextual attention, K=2) "
                     "segmentation head; GAP+MLP 3-class head; combined loss "
                     "L=0.4*CE(soft)+CE(final)+CE(cls); [grayscale,edge] 2-channel "
                     "input. Same grouped leakage-free 3-class split as the "
                     "classification baselines."),
        "config": {
            "img_size": D.IMG_SIZE, "epochs": args.epochs, "batch": args.batch,
            "lr": args.lr, "optimizer": "Adam", "scheduler": "cosine",
            "loss": "0.4*CE(soft)+CE(final)+CE(cls)", "alpha": ALPHA,
            "in_channels": 2, "K": 2, "params_M": params_m,
        },
        "class_names": list(D.CLASS_NAMES),
        "class_weights": [round(float(v), 6) for v in cls_w.cpu().tolist()],
        "split_sizes": {k: int((df.split == k).sum()) for k in ("train", "val", "test")},
        "seg_test_n": int(test_lesion.sum()),
        "best_val_dice": round(max(history["val_dice"]), 4),
        "best_val_macro_f1": round(max(history["val_macro_f1"]), 4),
        "train_minutes": train_minutes,
        "test_segmentation": seg,
        "test_detection": det,
        "test_classification": cls,
        "test": cls,
        "history": history,
    }
    R.save_results(name, result)
    R.plot_curve(history, name, keys=("train_loss", "val_dice", "val_macro_f1"))
    R.plot_confusion(cls["confusion_matrix"], list(D.CLASS_NAMES), name)
    print(f"[{name}]  test Dice={seg['dice']['mean']:.4f}  IoU={seg['iou']['mean']:.4f}  "
          f"|  macro-F1={cls['macro_f1']:.4f}  acc={cls['accuracy']:.4f}")
    print(f"  (paper reports Dice 0.8375 / Acc 0.9167 on BUSI-as-'OASBUD')")
    return result


if __name__ == "__main__":
    main()
