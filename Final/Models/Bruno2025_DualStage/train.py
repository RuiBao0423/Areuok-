# -*- coding: utf-8 -*-
"""
Reproduce Bruno et al. (2025) dual-stage pipeline [17] on the cleaned/grouped BUSI
benign+malignant subset, in the shared results format.

Pipeline (paper Fig. 1):
  Stage 1  DeepLabV3+ (reused from [07]) predicts the lesion mask.
  ROI      the predicted mask's bounding box crops a lesion ROI (model.crop_roi).
  Stage 2  EfficientNet-B0 (from [06]) classifies the ROI benign vs malignant.

Training: Stage-1 is reused (trained in [07]); we train ONLY Stage-2 on ROI crops.
The classifier is trained on GROUND-TRUTH-mask ROI crops (clean labels) and then the
FULL PIPELINE is evaluated end-to-end on the test set using PREDICTED-mask ROI crops
(the honest deployment number). We also report the GT-ROI classification as an
upper bound (perfect segmentation).

Metrics reported (paper's set):
  Segmentation: Dice, IoU, pixel accuracy   (Stage-1 on test)
  Classification: precision, recall, F1, accuracy, AUC   (2-class, pipeline + GT-ROI)

Run:  python Models/Bruno2025_DualStage/train.py
Outputs: Models/results/Bruno2025_DualStage.json (+ curve/confusion/ROI figures).
"""
import os, sys, time
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import albumentations as A

_HERE = os.path.dirname(os.path.abspath(__file__))
PROJ  = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, PROJ)
sys.path.insert(0, _HERE)
from Models.common import data as D
from Models.common import metrics as M
from Models.common import runner as R
from model import (build_seg_model, build_roi_classifier, roi_bbox_from_mask,
                   crop_roi, CLS_NAMES, LABEL2ID)

NAME    = "Bruno2025_DualStage"
EPOCHS  = 30
BATCH   = 16
LR      = 1e-4
ROI_SZ  = 256
CKPT    = os.path.join(_HERE, "roi_classifier_best.pt")


# ------------------------- ROI classification dataset ------------------------- #
def _aug():
    """Paper's augmentation (train only): rotate / translate / zoom / hflip."""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Affine(scale=(0.9, 1.1), translate_percent=0.05, rotate=(-15, 15),
                 border_mode=cv2.BORDER_CONSTANT, p=0.7),
        A.RandomBrightnessContrast(0.2, 0.2, p=0.3),
    ])


class ROICropDataset(Dataset):
    """In-memory 256x256 grayscale ROI crops + benign/malignant label.
    `crops` is a list of (uint8 HxW, label). Augment on train."""
    def __init__(self, crops, augment=False):
        self.crops = crops
        self.aug = _aug() if augment else None

    def __len__(self):
        return len(self.crops)

    def __getitem__(self, i):
        g, y = self.crops[i]
        if self.aug is not None:
            g = self.aug(image=g)["image"]
        x = D._norm_to_tensor3(g)                     # 3-ch ImageNet norm (pretrained EffNet)
        return x, torch.tensor(y, dtype=torch.long)


def build_gt_crops(df, split):
    """ROI crops from GROUND-TRUTH masks (for training/val the classifier)."""
    crops = []
    for _, row in df[df.split == split].iterrows():
        g, m = D.load_gray_and_mask(row)
        g256, m256, _ = D.letterbox(g, m)
        bbox = roi_bbox_from_mask(m256)
        crops.append((crop_roi(g256, bbox, ROI_SZ), LABEL2ID[row["cls"]]))
    return crops


@torch.no_grad()
def stage1_predict_test(seg_model, df):
    """Run Stage-1 on the test split. Returns per-image:
       gts (GT mask 256), preds (pred mask 256), gt_crop, pred_crop, label, example."""
    seg_model.eval()
    gts, preds, gt_crops, pred_crops, labels, examples = [], [], [], [], [], []
    for _, row in df[df.split == "test"].iterrows():
        g, m = D.load_gray_and_mask(row)
        g256, m256, _ = D.letterbox(g, m)
        x = D._norm_to_tensor3(g256).unsqueeze(0).to(R.DEVICE)
        pr = (torch.sigmoid(seg_model(x)).cpu().numpy()[0, 0] > 0.5).astype(np.uint8)
        gt = (m256 > 0).astype(np.uint8)
        y = LABEL2ID[row["cls"]]
        gts.append(gt); preds.append(pr); labels.append(y)
        gt_crops.append((crop_roi(g256, roi_bbox_from_mask(gt), ROI_SZ), y))
        pred_crops.append((crop_roi(g256, roi_bbox_from_mask(pr), ROI_SZ), y))
        if len(examples) < 6:
            examples.append((g256, gt, pr))
    return gts, preds, gt_crops, pred_crops, labels, examples


@torch.no_grad()
def classify(clf, crops):
    """Return (y_true, y_pred, y_proba) for a list of (crop, label)."""
    clf.eval()
    ds = ROICropDataset(crops, augment=False)
    ld = DataLoader(ds, BATCH, shuffle=False, num_workers=0)
    ys, ps, pr = [], [], []
    for x, y in ld:
        prob = torch.softmax(clf(x.to(R.DEVICE)), 1).cpu().numpy()
        ys.append(y.numpy()); ps.append(prob.argmax(1)); pr.append(prob)
    return np.concatenate(ys), np.concatenate(ps), np.concatenate(pr)


def save_roi_examples(examples, crops, name, n=6):
    """Grid: image+GT+pred mask (Stage-1) and the ROI crop fed to Stage-2."""
    import matplotlib
    matplotlib.use("Agg"); import matplotlib.pyplot as plt
    n = min(n, len(examples))
    fig, ax = plt.subplots(n, 4, figsize=(12, 3*n))
    if n == 1:
        ax = ax[None, :]
    for i in range(n):
        g, gt, pr = examples[i]
        crop = crops[i][0]
        ax[i, 0].imshow(g, cmap="gray")
        ax[i, 1].imshow(gt, cmap="gray")
        ax[i, 2].imshow(g, cmap="gray"); ax[i, 2].imshow(np.ma.masked_where(pr == 0, pr), cmap="autumn", alpha=.6)
        ax[i, 3].imshow(crop, cmap="gray")
        if i == 0:
            for a, t in zip(ax[i], ["image", "GT mask", "Stage-1 pred", "Stage-2 ROI crop"]):
                a.set_title(t)
        for a in ax[i]:
            a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"{name} - dual-stage pipeline (test set)"); fig.tight_layout()
    p = os.path.join(R.FIG_DIR, f"{name}_pipeline.png")
    fig.savefig(p, dpi=120); plt.close(fig)
    print(f"[runner] wrote {p}")
    return p


def main():
    R.set_seed(42)
    print(f"[{NAME}] device = {R.DEVICE}")
    df = D.make_split()                               # benign+malignant, leakage-free

    # ---------------- Stage 1: segmentation (reuse DeepLabV3+ [07]) ----------------
    seg_model, loaded = build_seg_model(load_checkpoint=True, device=R.DEVICE)
    print(f"[{NAME}] Stage-1 DeepLabV3+ checkpoint reused: {loaded}")
    gts, preds, gt_crops_te, pred_crops_te, labels_te, examples = stage1_predict_test(seg_model, df)
    seg = M.aggregate_seg(preds, gts)
    print(f"[{NAME}] Stage-1 test  Dice={seg['dice']['mean']}  IoU={seg['iou']['mean']}  pixAcc={seg['pixel_acc']['mean']}")

    # ---------------- Stage 2: ROI classifier (EfficientNet-B0 [06]) ----------------
    tr_crops = build_gt_crops(df, "train")
    va_crops = build_gt_crops(df, "val")
    print(f"[{NAME}] ROI crops -> train {len(tr_crops)}  val {len(va_crops)}  test {len(pred_crops_te)}")

    # inverse-frequency class weights (benign/malignant imbalance) -- Bruno oversample
    cnt = np.bincount([y for _, y in tr_crops], minlength=len(CLS_NAMES))
    w = cnt.sum() / (len(CLS_NAMES) * np.clip(cnt, 1, None))
    weight = torch.tensor(w, dtype=torch.float32, device=R.DEVICE)
    print(f"[{NAME}] train class counts {cnt.tolist()}  weights {w.round(3).tolist()}")

    clf = build_roi_classifier(num_classes=len(CLS_NAMES), pretrained=True).to(R.DEVICE)
    opt = torch.optim.Adam(clf.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss(weight=weight)
    tr = DataLoader(ROICropDataset(tr_crops, augment=True), BATCH, shuffle=True, num_workers=0)

    hist = {"train_loss": [], "val_f1": []}
    best = -1.0
    t0 = time.time()
    for ep in range(EPOCHS):
        clf.train(); losses = []
        for x, y in tr:
            x, y = x.to(R.DEVICE), y.to(R.DEVICE)
            opt.zero_grad(); loss = crit(clf(x), y); loss.backward(); opt.step()
            losses.append(loss.item())
        yv, pv, _ = classify(clf, va_crops)
        from sklearn.metrics import f1_score
        vf1 = float(f1_score(yv, pv, average="macro", labels=[0, 1], zero_division=0))
        hist["train_loss"].append(float(np.mean(losses))); hist["val_f1"].append(vf1)
        if vf1 > best:
            best = vf1; torch.save(clf.state_dict(), CKPT)
        if ep % 5 == 0 or ep == EPOCHS-1:
            print(f"  ep{ep:02d}  loss {np.mean(losses):.4f}  val_macroF1 {vf1:.4f}  (best {best:.4f})")
    train_min = (time.time() - t0) / 60

    # ---------------- end-to-end evaluation ----------------
    clf.load_state_dict(torch.load(CKPT, map_location=R.DEVICE))
    yt, pt, prt = classify(clf, pred_crops_te)        # FULL PIPELINE (predicted ROI)
    cls_pipe = M.classification_metrics(yt, pt, prt, CLS_NAMES)
    yg, pg, prg = classify(clf, gt_crops_te)          # upper bound (GT ROI)
    cls_gt = M.classification_metrics(yg, pg, prg, CLS_NAMES)

    # figures
    R.plot_curve(hist, NAME, keys=("train_loss", "val_f1"))
    R.plot_confusion(cls_pipe["confusion_matrix"], CLS_NAMES, NAME)
    save_roi_examples(examples, pred_crops_te, NAME)

    results = {
        "model": "Dual-stage pipeline: DeepLabV3+ (seg) -> ROI crop -> EfficientNet-B0 (cls)",
        "paper": "Bruno et al. (2025), Dual-stage framework [17]",
        "task": "segmentation-guided benign/malignant classification (2-class)",
        "our_work": "wrote the ROI-extraction coupling (mask bbox -> crop) and the end-to-end "
                    "pipeline training/eval; reused DeepLabV3+ [07] as Stage-1 and EfficientNet-B0 "
                    "[06] retrained on ROI crops as Stage-2; shared leakage-free grouped split",
        "config": {"img_size": D.IMG_SIZE, "roi_size": ROI_SZ, "epochs": EPOCHS, "batch": BATCH,
                   "lr": LR, "optimizer": "Adam", "loss": "weighted CrossEntropy",
                   "stage1": "DeepLabV3+ ResNet-50 (reused from [07])",
                   "stage2": "EfficientNet-B0 (ImageNet-pretrained, 2-class head)",
                   "classes": list(CLS_NAMES),
                   "stage1_checkpoint_reused": bool(loaded)},
        "split": {k: int((df.split == k).sum()) for k in ["train", "val", "test"]},
        "best_val_macro_f1": round(best, 4), "train_minutes": round(train_min, 2),
        "test_segmentation": seg,
        "test_classification": cls_pipe,               # PRIMARY: full-pipeline (predicted ROI)
        "test_classification_gt_roi": cls_gt,          # reference: perfect-segmentation upper bound
        "history": hist,
    }
    R.save_results(NAME, results)
    print(f"[{NAME}] PIPELINE test  seg Dice={seg['dice']['mean']}  IoU={seg['iou']['mean']}  || "
          f"cls macroF1={cls_pipe['macro_f1']}  acc={cls_pipe['accuracy']}  AUC={cls_pipe['macro_auc_ovr']}  "
          f"(GT-ROI F1={cls_gt['macro_f1']})  ({train_min:.1f} min)")


if __name__ == "__main__":
    main()
