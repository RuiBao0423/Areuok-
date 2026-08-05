# -*- coding: utf-8 -*-
"""
Models/ImprovedDualStage/train_strong.py

STRONGER Stage-2 for the Improved dual-stage pipeline. Same Stage-1 (Improved-UNet)
and same EfficientNet-B0 architecture, but a better-TRAINED classifier that fixes the
main weakness of the vanilla pipeline:

  * TRAIN/TEST DISTRIBUTION MATCH: the vanilla Stage-2 trains on clean GROUND-TRUTH-mask
    ROI crops but is tested on imperfect PREDICTED-mask ROI crops. Here we train the
    classifier on BOTH gt-mask AND predicted-mask ROI crops (2x data, matched to the
    test distribution).
  * more epochs (60) + cosine LR, model selection on PREDICTED-ROI val macro-F1.
  * classifier test-time augmentation (horizontal-flip averaging).

Everything is chosen on train/val only -- the test set is never used for tuning.
Writes Models/results/ImprovedDualStage_StrongCls.json.

Run:  python Models/ImprovedDualStage/train_strong.py
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
sys.path.insert(0, _HERE)
from Models.common import data as D
from Models.common import metrics as M
from Models.common import runner as R
from model import (build_seg_model, build_roi_classifier, roi_bbox_from_mask,
                   crop_roi, predict_mask, CLS_NAMES, LABEL2ID)
from train import ROICropDataset, build_gt_crops, save_roi_examples

NAME   = "ImprovedDualStage_StrongCls"
EPOCHS = 60
BATCH  = 16
LR     = 1e-4
ROI_SZ = 256
CKPT   = os.path.join(_HERE, "roi_classifier_strong.pt")


@torch.no_grad()
def build_pred_crops(seg_model, df, split, tta=False):
    """ROI crops from PREDICTED masks (Stage-1) for `split` -- matches test distribution."""
    seg_model.eval()
    crops, examples = [], []
    for _, row in df[df.split == split].iterrows():
        g, m = D.load_gray_and_mask(row)
        g256, m256, _ = D.letterbox(g, m)
        x = D._norm_to_tensor3(g256).unsqueeze(0).to(R.DEVICE)
        pr = predict_mask(seg_model, x, thresh=0.5, tta=tta, postprocess=True)[0]
        crops.append((crop_roi(g256, roi_bbox_from_mask(pr), ROI_SZ), LABEL2ID[row["cls"]]))
        if split == "test" and len(examples) < 6:
            examples.append((g256, (m256 > 0).astype(np.uint8), pr))
    return crops, examples


@torch.no_grad()
def classify_tta(clf, crops):
    """Predict with horizontal-flip TTA (average softmax of image and its mirror)."""
    clf.eval()
    ds = ROICropDataset(crops, augment=False)
    ld = DataLoader(ds, BATCH, shuffle=False, num_workers=0)
    ys, ps, pr = [], [], []
    for x, y in ld:
        x = x.to(R.DEVICE)
        p = torch.softmax(clf(x), 1) + torch.softmax(clf(torch.flip(x, dims=[-1])), 1)
        p = (p / 2).cpu().numpy()
        ys.append(y.numpy()); ps.append(p.argmax(1)); pr.append(p)
    return np.concatenate(ys), np.concatenate(ps), np.concatenate(pr)


def main():
    R.set_seed(42)
    print(f"[{NAME}] device = {R.DEVICE}")
    df = D.make_split()

    # Stage-1 (Improved-UNet) -- reuse checkpoint
    seg_model, loaded = build_seg_model(load_checkpoint=True, device=R.DEVICE)
    print(f"[{NAME}] Stage-1 Improved-UNet reused: {loaded}")

    # segmentation quality on test (same Stage-1 -> same as ImprovedDualStage)
    pred_test, examples = build_pred_crops(seg_model, df, "test", tta=True)
    gts, preds = [], []
    for _, row in df[df.split == "test"].iterrows():
        g, m = D.load_gray_and_mask(row); g256, m256, _ = D.letterbox(g, m)
        x = D._norm_to_tensor3(g256).unsqueeze(0).to(R.DEVICE)
        preds.append(predict_mask(seg_model, x, tta=True, postprocess=True)[0])
        gts.append((m256 > 0).astype(np.uint8))
    seg = M.aggregate_seg(preds, gts)
    gt_test = build_gt_crops(df, "test")

    # ---- Stage-2 training data: GT-ROI + PREDICTED-ROI crops (distribution match) ----
    gt_tr = build_gt_crops(df, "train"); gt_va = build_gt_crops(df, "val")
    pred_tr, _ = build_pred_crops(seg_model, df, "train"); pred_va, _ = build_pred_crops(seg_model, df, "val")
    tr_crops = gt_tr + pred_tr                       # 2x data, both clean and realistic
    va_crops = pred_va                               # select on predicted-ROI val (test-matched)
    print(f"[{NAME}] Stage-2 train crops {len(tr_crops)} (gt {len(gt_tr)} + pred {len(pred_tr)})  val {len(va_crops)}")

    cnt = np.bincount([y for _, y in tr_crops], minlength=len(CLS_NAMES))
    w = cnt.sum() / (len(CLS_NAMES) * np.clip(cnt, 1, None))
    weight = torch.tensor(w, dtype=torch.float32, device=R.DEVICE)

    clf = build_roi_classifier(num_classes=len(CLS_NAMES), pretrained=True).to(R.DEVICE)
    opt = torch.optim.Adam(clf.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    crit = nn.CrossEntropyLoss(weight=weight)
    tr = DataLoader(ROICropDataset(tr_crops, augment=True), BATCH, shuffle=True, num_workers=0)

    from sklearn.metrics import f1_score
    hist = {"train_loss": [], "val_f1": []}; best = -1.0
    t0 = time.time()
    for ep in range(EPOCHS):
        clf.train(); losses = []
        for x, y in tr:
            x, y = x.to(R.DEVICE), y.to(R.DEVICE)
            opt.zero_grad(); loss = crit(clf(x), y); loss.backward(); opt.step()
            losses.append(loss.item())
        sched.step()
        yv, pv, _ = classify_tta(clf, va_crops)
        vf1 = float(f1_score(yv, pv, average="macro", labels=[0, 1], zero_division=0))
        hist["train_loss"].append(float(np.mean(losses))); hist["val_f1"].append(vf1)
        if vf1 > best:
            best = vf1; torch.save(clf.state_dict(), CKPT)
        if ep % 10 == 0 or ep == EPOCHS-1:
            print(f"  ep{ep:02d}  loss {np.mean(losses):.4f}  val_macroF1 {vf1:.4f}  (best {best:.4f})")
    train_min = (time.time() - t0) / 60

    clf.load_state_dict(torch.load(CKPT, map_location=R.DEVICE))
    yt, pt, prt = classify_tta(clf, pred_test)       # FULL PIPELINE (predicted ROI + TTA)
    cls_pipe = M.classification_metrics(yt, pt, prt, CLS_NAMES)
    yg, pg, prg = classify_tta(clf, gt_test)         # upper bound (GT ROI)
    cls_gt = M.classification_metrics(yg, pg, prg, CLS_NAMES)

    R.plot_curve(hist, NAME, keys=("train_loss", "val_f1"))
    R.plot_confusion(cls_pipe["confusion_matrix"], CLS_NAMES, NAME)
    save_roi_examples(examples, pred_test, NAME)

    results = {
        "model": "Improved dual-stage + strengthened Stage-2 (Improved-UNet -> ROI -> EfficientNet-B0*)",
        "paper": "Our extension of Bruno et al. (2025) [17]; Stage-2 trained on gt+predicted ROI, TTA",
        "task": "segmentation-guided benign/malignant classification (2-class)",
        "our_work": "Stage-2 EfficientNet-B0 trained on BOTH gt-mask and predicted-mask ROI crops "
                    "(train/test distribution match, 2x data), 60 epochs cosine, val selection on "
                    "predicted-ROI, horizontal-flip test-time augmentation. Stage-1 = Improved-UNet (reused).",
        "config": {"img_size": D.IMG_SIZE, "roi_size": ROI_SZ, "epochs": EPOCHS, "batch": BATCH,
                   "lr": LR, "optimizer": "Adam", "scheduler": "cosine", "loss": "weighted CrossEntropy",
                   "stage1": "Improved Pretrained-UNet (reused)",
                   "stage2": "EfficientNet-B0 trained on gt+pred ROI, hflip TTA",
                   "train_data": "gt-ROI + predicted-ROI crops", "classes": list(CLS_NAMES),
                   "stage1_checkpoint_reused": bool(loaded)},
        "split": {k: int((df.split == k).sum()) for k in ["train", "val", "test"]},
        "best_val_macro_f1": round(best, 4), "train_minutes": round(train_min, 2),
        "test_segmentation": seg,
        "test_classification": cls_pipe,
        "test_classification_gt_roi": cls_gt,
        "history": hist,
        "baselines": {"bruno_pipeline_f1": 0.804, "improved_pipeline_f1": 0.807,
                      "standalone_efficientnet_3class_f1": 0.828},
    }
    R.save_results(NAME, results)
    print(f"[{NAME}] PIPELINE test  seg Dice={seg['dice']['mean']}  || cls macroF1={cls_pipe['macro_f1']}  "
          f"acc={cls_pipe['accuracy']}  AUC={cls_pipe['macro_auc_ovr']}  (GT-ROI F1={cls_gt['macro_f1']})  ({train_min:.1f} min)")
    print(f"[{NAME}] vs: Bruno 0.804 | Improved-vanilla 0.807 | standalone-EffNet(3cls) 0.828")


if __name__ == "__main__":
    main()
