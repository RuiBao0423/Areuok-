# CResU-Net [13] — source & attribution

**Paper reproduced:** Derakhshandeh & Mahloojifar, *Modifying the U-Net's
Encoder-Decoder Architecture for Segmentation of Tumors in Breast Ultrasound Images*
(2024). Highest reported BUSI Dice in the surveyed literature: **Dice 82.88 ± 3.1,
IoU 77.5 ± 2.9** (5-fold CV, 8.88 M params).

**Naming note:** the "C" is the **Co-Block** (a dense-**concatenation** residual
block), *not* a "contextual"/attention module — the paper has no attention/ASPP/SE.

## What is **our own work**
- `model.py` — the three signature blocks and the block-scheduled U-Net, from scratch
  in PyTorch (paper is TensorFlow/Keras):
  - **Co-Block**: three 3×3 convs (F, 2F, 4F) with **dense concatenations** → 7F ch;
  - **ResNet-identity** block (1×1→3×3→1×1 + shortcut);
  - **MultiRes** block (3 stacked convs concatenated + 1×1 residual);
  - encoder block schedule **E1,E2,E4 = Co-Block; E3 = ResIdentity; E5 = MultiRes**;
    dropout in E5 + bottleneck; pure **Dice loss**.
- `train.py` — full training/eval pipeline, metric/JSON/figure export.
- `../common/` — shared cleaning, grouped **leakage-free split**, letterbox, metrics.

## Faithful to the paper
The three block types + the per-stage encoder block schedule (the key contribution);
pure Dice loss; 256² input; base filter counts [16,32,64,128,256]/512.

## Deviations (documented, deliberate)
| Paper | Ours | Why |
|---|---|---|
| 5-fold CV, trained 5×, mean±std | single grouped **leakage-free** split | our honest split protocol; comparable to the other models |
| double skip (`O_E^i` + post-pool `O_ME^{i-1}`) | single U-Net skip | paper gives no channel table; single skip keeps it correct/runnable |
| decoder mirrors the per-stage block types | decoder uses **Co-Blocks uniformly** | simplification; the Co-Block is the novelty and is preserved |
| NLM speckle denoise + 180° rotation aug | repo letterbox + augmentation | identical preprocessing to the baselines |
| optimizer/lr/batch unspecified | Adam 1e-3, batch 8, 120 epochs | filled the unspecified gaps |
| 8.88 M params (exact channel table not given) | ~similar scale, not exact | per-layer channels unspecified; 1×1 projections used to control width |

Run: `python Models/CResUNet_Derakhshandeh2024/train.py` → writes `Models/results/CResUNet_Derakhshandeh2024.json`.
