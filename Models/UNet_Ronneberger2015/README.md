# U-Net [03] — source & attribution

**Paper reproduced:** Ronneberger, Fischer & Brox, *U-Net: Convolutional Networks for
Biomedical Image Segmentation*, MICCAI 2015. DOI 10.1007/978-3-319-24574-4_28 (arXiv:1505.04597).
Canonical reference implementation (for cross-checking the architecture only):
<https://github.com/milesial/Pytorch-UNet> and the authors' original Caffe code.

## What is **our own work**
- `model.py` — the U-Net architecture written from scratch in PyTorch (encoder/decoder +
  skip connections, `DoubleConv` blocks) following Fig. 1 of the paper.
- `train.py` — the entire training/evaluation pipeline: Dice+BCE loss, Adam + cosine LR,
  checkpointing, test-set inference, metric computation and figure/JSON export.
- `../common/data.py` — data **cleaning** (drop the exact cross-class duplicate),
  multi-mask **union**, **letterbox** preprocessing, augmentation, and the **grouped,
  leakage-free split** (all driven by our EDA).
- `../common/metrics.py` — Dice / IoU / pixel metrics.

## Deviations from the 2015 paper (deliberate, for this small BUSI task)
| Paper (2015) | Ours | Why |
|---|---|---|
| unpadded 3×3 convs + cropping | padded 3×3 convs (same-size output) | simpler, avoids border cropping on 256² inputs |
| no BatchNorm | BatchNorm after each conv | standard modern stabiliser for small data |
| weighted pixel-wise cross-entropy | Dice + BCE | EDA 3.5 shows many tiny lesions → Dice handles pixel imbalance |
| SGD, momentum 0.99 | Adam, lr 1e-3, cosine | faster convergence on a small set |
| input 572², EM/cell data | input 256², BUSI ultrasound | our dataset |

Run: `python Models/UNet_Ronneberger2015/train.py` → writes `Models/results/UNet_Ronneberger2015.json`.
