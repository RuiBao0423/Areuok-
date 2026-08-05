# Patch-based LeNet [02] — source & attribution

**Paper reproduced:** Yap et al., *Automated Breast Ultrasound Lesions Detection Using
Convolutional Neural Networks*, IEEE J. Biomedical and Health Informatics, 22(4), 2018.
DOI 10.1109/JBHI.2017.2731873. We reproduce the paper's **Patch-based LeNet** method
(their Fig. 3 architecture and Section IV-A/B pipeline). The paper has no official public
code; our implementation follows the textual/figure description in the paper.

## What is **our own work**
- `model.py` — the LeNet patch classifier (conv5×5-20 → pool → conv5×5-50 → pool →
  FC500 → FC2) written from scratch in PyTorch, matching Fig. 3.
- `train.py` — the full pipeline we wrote: balanced lesion/non-lesion **patch sampling**,
  RMSprop training with per-epoch resampling, **sliding-window inference** that builds a
  lesion-probability map, thresholding + small-region removal, and metric/figure/JSON export.
- `../common/data.py`, `../common/metrics.py` — shared cleaning, split, and metrics
  (incl. the paper's **TPF / FPs-per-image / F-measure** detection metrics, eq. 11–13).

## Faithful to the paper
- 28×28 grayscale patches; lesion vs non-lesion two-class problem; LeNet architecture;
  RMSprop (lr 0.01) with dropout 0.33; sliding-window testing; removal of tiny predicted
  regions; evaluation with TPF, FPs/image and F-measure.

## Deviations (deliberate)
| Paper (2018) | Ours | Why |
|---|---|---|
| Datasets A/B (306 + 163 imgs), detection-only | BUSI benign+malignant (645), + Dice/IoU too | our dataset; adds overlap metrics for comparison with U-Net |
| Caffe, 10-fold CV | PyTorch, single grouped split | our stack; leakage-free split from EDA |
| seed point = region centre vs GT ROI box | same rule (centroid ∈ GT bbox) | kept faithful |

Run: `python Models/Yap2018_PatchLeNet/train.py` → writes `Models/results/Yap2018_PatchBasedLeNet.json`.
