# -*- coding: utf-8 -*-
"""
Train + evaluate ResNet [04] (He et al., CVPR 2016) for 3-class BUSI classification
(normal / benign / malignant) on the cleaned/grouped, leakage-free split, then write a
metrics JSON. Transfer learning: ImageNet-pretrained ResNet-18 with a fresh 3-way head,
fine-tuned with SGD (momentum 0.9, weight decay 1e-4) as in the paper's training recipe.
Class imbalance (normal/benign/malignant) is handled with inverse-frequency CrossEntropy.

Primary metric: macro-F1 (imbalance-fair); we also report per-class P/R/F1, balanced acc,
accuracy, macro one-vs-rest AUC and the confusion matrix (see Evaluation_Metrics.md).

Run:  python Models/ResNet_He2016/train.py
Outputs: Models/results/ResNet_He2016.json (+ curve/confusion figures, checkpoint).
"""
import os, sys, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

_HERE = os.path.dirname(os.path.abspath(__file__))
PROJ  = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, PROJ)
from Models.common import data as D
from Models.common import metrics as M
from Models.common import runner as R
from Models.ResNet_He2016.model import ResNetClassifier

NAME   = "ResNet_He2016"
ARCH   = "resnet18"    # BUSI is tiny -> ResNet-18; "resnet50" available in model.py
EPOCHS = 60
BATCH  = 16
LR     = 1e-3          # fine-tuning LR (paper trained from scratch at 0.1; we transfer)
CKPT   = os.path.join(_HERE, "best.pt")


@torch.no_grad()
def predict(model, loader):
    """Return (y_true, y_pred, y_proba) over a loader."""
    model.eval(); ys, ps, probs = [], [], []
    for x, y in loader:
        prob = torch.softmax(model(x.to(R.DEVICE)), dim=1).cpu().numpy()
        ys.append(y.numpy()); ps.append(prob.argmax(1)); probs.append(prob)
    return np.concatenate(ys), np.concatenate(ps), np.concatenate(probs)


def main():
    R.set_seed(42)
    df = D.make_cls_split()
    tr = DataLoader(D.ClsDataset(df, "train", augment=True, rgb3=True), BATCH, shuffle=True, num_workers=0)
    va = DataLoader(D.ClsDataset(df, "val",   rgb3=True), BATCH, shuffle=False, num_workers=0)

    model = ResNetClassifier(len(D.CLASS_NAMES), arch=ARCH, pretrained=True).to(R.DEVICE)
    weight = D.class_weights(df, "train").to(R.DEVICE)
    crit = nn.CrossEntropyLoss(weight=weight)
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    hist = {"train_loss": [], "val_f1": []}
    best = -1
    print(f"[{NAME}] device={R.DEVICE}  arch={ARCH}  train={len(tr.dataset)} val={len(va.dataset)}")
    print(f"[{NAME}] class weights (normal/benign/malignant) = {weight.cpu().numpy().round(3)}")
    t0 = time.time()
    for ep in range(EPOCHS):
        model.train(); losses = []
        for x, y in tr:
            x, y = x.to(R.DEVICE), y.to(R.DEVICE)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward(); opt.step()
            losses.append(loss.item())
        sched.step()
        yv, pv, _ = predict(model, va)
        vf1 = float(f1_score(yv, pv, average="macro", labels=list(range(len(D.CLASS_NAMES))), zero_division=0))
        hist["train_loss"].append(float(np.mean(losses))); hist["val_f1"].append(vf1)
        if vf1 > best:
            best = vf1; torch.save(model.state_dict(), CKPT)
        if ep % 5 == 0 or ep == EPOCHS-1:
            print(f"  ep{ep:02d}  loss {np.mean(losses):.4f}  val_macroF1 {vf1:.4f}  (best {best:.4f})")
    train_min = (time.time()-t0)/60

    # ---- test with best checkpoint ----
    model.load_state_dict(torch.load(CKPT, map_location=R.DEVICE)); model.eval()
    te = DataLoader(D.ClsDataset(df, "test", rgb3=True), BATCH, shuffle=False, num_workers=0)
    yt, pt, prt = predict(model, te)
    cls = M.classification_metrics(yt, pt, prt, D.CLASS_NAMES)

    R.plot_curve(hist, NAME, keys=("train_loss", "val_f1"))
    R.plot_confusion(cls["confusion_matrix"], D.CLASS_NAMES, NAME)

    results = {
        "model": f"ResNet ({ARCH}, transfer learning)",
        "paper": "He et al. (2016), CVPR [04]",
        "task": "3-class classification (normal / benign / malignant)",
        "our_work": "torchvision ImageNet-pretrained ResNet with a fresh 3-way head; "
                    "inverse-frequency CrossEntropy; letterbox + ImageNet-norm preprocessing; "
                    "grouped leakage-free split shared with the other baselines",
        "config": {"img_size": D.IMG_SIZE, "arch": ARCH, "epochs": EPOCHS, "batch": BATCH, "lr": LR,
                   "optimizer": "SGD(momentum=0.9, wd=1e-4)", "loss": "weighted CrossEntropy",
                   "params_M": round(sum(p.numel() for p in model.parameters())/1e6, 2)},
        "split": {k: int((df.split == k).sum()) for k in ["train", "val", "test"]},
        "best_val_macro_f1": round(best, 4), "train_minutes": round(train_min, 2),
        "test_classification": cls,
        "history": hist,
    }
    R.save_results(NAME, results)
    print(f"[{NAME}] TEST  macroF1={cls['macro_f1']}  balAcc={cls['balanced_accuracy']}  "
          f"acc={cls['accuracy']}  macroAUC={cls['macro_auc_ovr']}  "
          f"malignant_recall={cls['per_class']['malignant']['recall']}  ({train_min:.1f} min)")


if __name__ == "__main__":
    main()
