# AAU-Net [12] — source & attribution

**Paper reproduced:** Chen et al., *AAU-net: An Adaptive Attention U-net for Breast
Lesions Segmentation in Ultrasound Images*, IEEE TMI (2022/2023). Official code:
<https://github.com/CGPxy/AAU-net>. Reported on **BUSI (4-fold CV): Dice 77.51 ± 0.68,
IoU/Jaccard 68.82 ± 0.44**.

## What is **our own work**
- `model.py` — the **HAAM (Hybrid Adaptive Attention Module)** and the U-Net that uses
  it, written from scratch in PyTorch (paper is TensorFlow):
  - three parallel multi-receptive-field convs (3×3, 5×5, dilated-3×3 rate 3);
  - **channel self-attention** producing `α`; the complement `1−α` adaptively routes
    the two large-receptive-field branches (`FD_C=α·FD`, `FS_C=(1−α)·F5`);
  - **spatial self-attention** producing `β` (and `1−β`) to fuse location info → `F_out`;
  - two stacked HAAMs per encoder/decoder stage (5 levels, [32,64,128,256,512]).
- `train.py` — full training/eval pipeline, metric/JSON/figure export.
- `../common/` — shared cleaning, grouped **leakage-free split**, letterbox, metrics.

## Faithful to the paper
HAAM structure (multi-kernel + channel α/1−α routing + spatial β/1−β), two HAAMs per
stage, U-Net skeleton with 4 down/4 up, Adam lr 1e-3.

## Deviations (documented, deliberate)
| Paper | Ours | Why |
|---|---|---|
| BCE loss only | **Dice + BCE** | repo convention; more stable, comparable to the other seg baselines |
| 4-fold cross-validation | single grouped **leakage-free** split | our honest split protocol; comparable to the other models |
| input size unstated, no augmentation described | 256² letterbox + repo augmentation | fills unspecified gaps; identical preprocessing to the baselines |
| 50 epochs, batch 12 | 80 epochs, batch 8 | small tuning for our single-split setup |
| TensorFlow | PyTorch (LeakyReLU slope 0.2) | framework port |

Run: `python Models/AAUNet_Chen2023/train.py` → writes `Models/results/AAUNet_Chen2023.json`.
