# ResNet [04] — source & attribution

**Paper reproduced:** He, Zhang, Ren & Sun, *Deep Residual Learning for Image Recognition*,
CVPR 2016 (arXiv:1512.03385). Winner of ILSVRC-2015 classification.
Canonical reference implementation (used only for the architecture / pretrained weights):
`torchvision.models.resnet18` / `resnet50` with `IMAGENET1K_V1` weights.

**Task:** 3-class breast-ultrasound classification — **normal / benign / malignant** — on the
full cleaned BUSI set (all three classes, unlike the segmentation baselines which only use the
647 benign+malignant lesion images).

## What is **our own work**
- `model.py` — the transfer-learning wrapper: load an ImageNet-pretrained ResNet (the residual
  architecture of the paper) and replace its 1000-way fc head with a 3-way linear head.
- `train.py` — the whole training/evaluation pipeline: inverse-frequency weighted CrossEntropy,
  SGD + cosine LR, checkpointing on best validation macro-F1, test-set inference, metric
  computation and figure/JSON export.
- `../common/data.py` — `make_cls_split()` (all-3-class, grouped **leakage-free** split, cached
  to `split_cls.csv`) and `ClsDataset` (letterbox + ImageNet-norm, image-only augmentation).
- `../common/metrics.py` — `classification_metrics()` (macro-F1, per-class P/R/F1, balanced acc,
  accuracy, macro one-vs-rest AUC, confusion matrix).

## Deviations from the 2016 paper (deliberate, for this small BUSI task)
| Paper (2016) | Ours | Why |
|---|---|---|
| trained from scratch on ImageNet (1.28M imgs) | ImageNet-pretrained, fine-tuned | BUSI has ~780 images — transfer learning is essential |
| ResNet-50/101/152 headline results | ResNet-18 default | tiny dataset → smallest variant resists overfitting (`arch="resnet50"` available) |
| SGD lr 0.1, /10 on plateau, 60×10⁴ iters | SGD (mom 0.9, wd 1e-4) lr 1e-3, cosine, 60 epochs | fine-tuning, not from-scratch; cosine is a modern default |
| plain CrossEntropy, balanced ImageNet | **inverse-frequency weighted** CrossEntropy | normal/benign/malignant are imbalanced |
| 224² crops, 1000 classes | 256² letterbox, 3 classes | our preprocessing / task |
| RGB natural images | grayscale replicated to 3-ch (pseudo-RGB) | ultrasound is single-channel; lets pretrained conv1 transfer |

## Split note
The segmentation baselines call `make_split()` → `split.csv` (benign+malignant only).
Classification calls `make_cls_split()` → **`split_cls.csv`** (all three classes, adds an integer
`label` column: normal=0, benign=1, malignant=2). Both use identical cleaning (drop the exact
cross-class duplicate) and the same pHash-clustered, grouped, stratified 70/15/15 split, so no
near-duplicate leaks across train/val/test.

Run: `python Models/ResNet_He2016/train.py` → writes `Models/results/ResNet_He2016.json`.
