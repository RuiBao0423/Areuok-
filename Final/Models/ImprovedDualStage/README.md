# Improved Dual-Stage Pipeline — our extension of Bruno [17]

**What it is:** the same segmentation-guided classification pipeline as Bruno et al.
(2025) [17], but with a **stronger Stage-1 segmenter**. It answers the question raised
by the Bruno reproduction — *is the pipeline limited by segmentation or by the
classifier?* — by controlled substitution.

```
Stage 1 (segmentation)              ROI                Stage 2 (classification)
Improved Pretrained-UNet    ─►  crop predicted   ─►   EfficientNet-B0 : benign vs malignant
(EffB4 + Focal Tversky +        mask's bbox
 TTA + largest-CC)
```

## What is **our own work**
- `model.py` — loads our **Improved Pretrained-UNet** (Dice 0.759) as Stage-1 and
  reuses Bruno's ROI coupling + Stage-2 EfficientNet-B0 classifier **unchanged**.
- `train.py` — the controlled experiment: identical to `Bruno2025_DualStage/train.py`
  except Stage-1. Stage-1 masks use the Improved model's **TTA + post-processing**;
  Stage-2 is trained on GT-mask ROI crops and the full pipeline is evaluated on
  predicted-mask ROI crops (the honest number), plus a GT-ROI upper bound.

## Controlled comparison (only Stage-1 changes)
| | Stage-1 segmenter | Stage-1 Dice | Pipeline macro-F1 | GT-ROI macro-F1 |
|---|---|---|---|---|
| **Bruno [17]** | DeepLabV3+ (ResNet-50) | 0.739 | 0.804 | 0.865 |
| **Ours** | Improved Pretrained-UNet | **0.759** | *(see results)* | *(see results)* |

Everything except Stage-1 (ROI extraction, Stage-2 classifier, split, augmentation,
metrics) is byte-for-byte reused from Bruno, so any change in the pipeline number is
attributable to the improved segmentation alone.

## Prerequisite
Stage-1 reuses the checkpoint saved by `Models/Improvement/train.py`
(`Models/Improvement/improved_unet_best.pt`). Run that first if the checkpoint is absent.

Run: `python Models/ImprovedDualStage/train.py` → writes `Models/results/ImprovedDualStage.json`.
