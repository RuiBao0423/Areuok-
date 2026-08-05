# nnU-Net [20] — source & attribution

**Paper reproduced:** Isensee, Jaeger, Kohl, Petersen & Maier-Hein, *nnU-Net: a
self-configuring method for deep learning-based biomedical image segmentation*,
Nature Methods 18:203–211 (2021). arXiv:1904.08128.

nnU-Net has **no breast-ultrasound experiment** (its 19 datasets are MRI/CT/EM only),
so this is an *architecture-and-recipe* reproduction applied to BUSI, not a number-for-
number reproduction. We reproduce nnU-Net's fixed **"blueprint" design** — the part
that is a deterministic template, not the self-configuring heuristics.

## What is **our own work**
- `model.py` — the 2D nnU-Net blueprint written from scratch in PyTorch:
  2×(Conv3×3 → **InstanceNorm** → **LeakyReLU 0.01**) per stage, **strided-conv**
  downsampling, **transposed-conv** upsampling, base 32 features doubled and capped at
  512, and **deep supervision** heads on the upper decoder resolutions.
- `train.py` — the full training/eval pipeline: **CE + soft-Dice, deep-supervised**
  across resolutions with weights [1, ½, ¼] (normalised); **SGD Nesterov 0.99**,
  weight decay 3e-5, **polyLR** `lr=0.01·(1−t/T)^0.9`; test inference + metric/JSON/figure export.
- `../common/` — the shared cleaning, grouped **leakage-free split**, letterbox
  preprocessing and metrics (identical to the other seg baselines → comparable).

## Faithful (blueprint parameters)
InstanceNorm; LeakyReLU 0.01; 2 conv blocks/stage; strided-conv down / transposed-conv
up; 32→512 feature rule; deep supervision; CE+Dice loss; SGD-Nesterov 0.99 / lr 0.01 /
polyLR^0.9.

## Deviations from full nnU-Net (documented, deliberate)
| Full nnU-Net | Ours | Why |
|---|---|---|
| dataset fingerprinting → auto patch/spacing/batch/depth | hard-coded 256², 5 stages, batch 8 | no self-configuration; single fixed 2D config |
| z-score per image + `batchgenerators` augmentation | repo letterbox + [-1,1] + repo augmentation | keeps split/preprocessing **identical to the other seg baselines** for a fair comparison |
| 1000 epochs × 250 iters (250k steps) | fixed 200 epochs | compute budget; documented |
| 2D + 3D full-res + 3D low-res + cascade, 5-fold CV, ensembling | single 2D model, one split | scope |
| sliding-window tiled inference + TTA mirroring | whole-image inference, no TTA | isolates the architecture's contribution |
| empirical largest-CC postprocessing (if CV shows gain) | none | pure architecture comparison |

Run: `python Models/nnUNet_Isensee2021/train.py` → writes `Models/results/nnUNet_Isensee2021.json`.
