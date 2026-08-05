# -*- coding: utf-8 -*-
"""
Train + evaluate FCN-AlexNet [02] (Yap's transfer-learning method, their best) on the
cleaned/grouped BUSI split. Faithful to the paper: ImageNet-pretrained AlexNet made fully
convolutional, fine-tuned with SGD (lr 0.001, momentum, 60 epochs, dropout 0.33).
It is Yap's best DETECTION method, so we report TPF/FPs/F-measure (vs the paper) AND
Dice/IoU (so it can be compared with U-Net on segmentation).
Run:  python Models/Yap2018_FCN_AlexNet/train.py
Outputs: Models/results/Yap2018_FCN_AlexNet.json (+ figures, checkpoint).
"""
import os, sys, time
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
from Models.Yap2018_FCN_AlexNet.model import FCNAlexNet

NAME   = "Yap2018_FCN_AlexNet"
EPOCHS = 60
BATCH  = 8
LR     = 1e-3          # paper: SGD lr 0.001
CKPT   = os.path.join(_HERE, "best.pt")
# Yap et al. (2018) Table I reported detection results for FCN-AlexNet (their best):
PAPER = {"FCN-AlexNet_A": {"TPF": 0.98, "FPs_per_image": 0.16, "F_measure": 0.91},
         "FCN-AlexNet_B": {"TPF": 0.92, "FPs_per_image": 0.17, "F_measure": 0.89}}


def dice_bce_loss(logits, target):
    bce = nn.functional.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    inter = (p*target).sum((1, 2, 3))
    dl = 1 - (2*inter + 1) / (p.sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1)
    return bce + dl.mean()


@torch.no_grad()
def val_dice(model, loader):
    model.eval(); ds = []
    for x, y in loader:
        p = torch.sigmoid(model(x.to(R.DEVICE))).cpu().numpy() > 0.5
        for pi, yi in zip(p, y.numpy()):
            ds.append(M.dice(pi[0], yi[0] > 0.5))
    return float(np.mean(ds))


def main():
    R.set_seed(42)
    df = D.make_split()
    tr = DataLoader(D.SegDataset(df, "train", augment=True, rgb3=True), BATCH, shuffle=True)
    va = DataLoader(D.SegDataset(df, "val",   rgb3=True), BATCH, shuffle=False)

    model = FCNAlexNet(1, dropout=0.33, pretrained=True).to(R.DEVICE)
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    hist = {"train_loss": [], "val_dice": []}
    best = -1
    print(f"[{NAME}] device={R.DEVICE}  train={len(tr.dataset)} val={len(va.dataset)}")
    t0 = time.time()
    for ep in range(EPOCHS):
        model.train(); losses = []
        for x, y in tr:
            x, y = x.to(R.DEVICE), y.to(R.DEVICE)
            opt.zero_grad(); loss = dice_bce_loss(model(x), y); loss.backward(); opt.step()
            losses.append(loss.item())
        sched.step()
        vd = val_dice(model, va)
        hist["train_loss"].append(float(np.mean(losses))); hist["val_dice"].append(vd)
        if vd > best:
            best = vd; torch.save(model.state_dict(), CKPT)
        if ep % 5 == 0 or ep == EPOCHS-1:
            print(f"  ep{ep:02d}  loss {np.mean(losses):.4f}  val_dice {vd:.4f}  (best {best:.4f})")
    train_min = (time.time()-t0)/60

    model.load_state_dict(torch.load(CKPT, map_location=R.DEVICE)); model.eval()
    preds, gts, examples = [], [], []
    with torch.no_grad():
        for row, g, m, v in D.iter_test_images(df, "test"):
            x = D._norm_to_tensor3(g).unsqueeze(0).to(R.DEVICE)
            pr = (torch.sigmoid(model(x)).cpu().numpy()[0, 0] > 0.5).astype(np.uint8)
            preds.append(pr); gts.append((m > 0).astype(np.uint8))
            if len(examples) < 6:
                examples.append((g, (m > 0).astype(np.uint8), pr))

    seg = M.aggregate_seg(preds, gts)
    det = M.detection_metrics(preds, gts)
    R.plot_curve(hist, NAME, keys=("train_loss", "val_dice"))
    R.save_seg_examples(examples, NAME)

    results = {
        "model": "FCN-AlexNet (transfer learning)", "paper": "Yap et al. (2018), IEEE JBHI [02] (their best method)",
        "task": "lesion detection/segmentation via transfer-learned fully convolutional AlexNet (benign+malignant)",
        "our_work": "from-scratch PyTorch FCN head on torchvision ImageNet-pretrained AlexNet backbone; "
                    "Dice+BCE; grouped leakage-free split; shared metrics",
        "config": {"img_size": D.IMG_SIZE, "epochs": EPOCHS, "batch": BATCH, "lr": LR,
                   "optimizer": "SGD(momentum=0.9)", "dropout": 0.33, "backbone": "AlexNet(ImageNet)",
                   "params_M": round(sum(p.numel() for p in model.parameters())/1e6, 2)},
        "split": {k: int((df.split == k).sum()) for k in ["train", "val", "test"]},
        "best_val_dice": round(best, 4), "train_minutes": round(train_min, 2),
        "test_segmentation": seg, "test_detection": det,
        "paper_reported_detection": PAPER,
        "history": hist,
    }
    R.save_results(NAME, results)
    print(f"[{NAME}] TEST  Dice={seg['dice']['mean']}  IoU={seg['iou']['mean']}  "
          f"| detection F={det['F_measure']} TPF={det['TPF']} FPs/img={det['FPs_per_image']}  ({train_min:.1f} min)")


if __name__ == "__main__":
    main()
