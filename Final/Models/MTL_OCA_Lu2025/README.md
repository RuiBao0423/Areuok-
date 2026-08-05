# MTL-OCA [25] — source & attribution

**Paper reproduced:** Lu et al., *Automatic joint segmentation and classification of
breast ultrasound images via multi-task learning with object contextual attention*,
Frontiers in Oncology 15:1567577 (2025). Model name: **MTL-OCA**.

**Dataset note (important):** the paper calls its primary dataset "OASBUD", but its
signature — **780 images, 600 women, ages 25–75, benign 437 / malignant 210 / normal
133** — is *exactly BUSI* (Al-Dhabyani 2020), not the real OASBUD (~100 RF images).
So their primary benchmark IS BUSI, and we reproduce on BUSI. Their reported BUSI
numbers: **Dice 83.75 %, IoU 72.03 %, classification Acc 91.67 %**.

## What is **our own work**
- `model.py` — written from scratch:
  - **Res-UNet** shared backbone (residual blocks + **GroupNorm**), feeding both tasks.
  - **OCA** = Object Contextual Attention (OCR-style, **K=2** object regions): a coarse
    1×1 head produces soft object regions (`L_soft`); OCA aggregates region features,
    computes pixel↔region attention, and augments each pixel with its object context to
    produce the final mask (`L_aug`) — Eqs. 1–3 of the paper.
  - classification head = GAP over the backbone bottleneck → 2-layer MLP → 3-way.
  - `MTLDataset` — yields a 2-channel `[grayscale, Gaussian-derivative edge]` input +
    binary mask + 3-class label, for all three classes (normal has an all-black mask).
- `train.py` — combined loss `L = 0.4·CE(soft) + CE(final) + CE(cls)` (α=0.4, paper),
  Adam lr 1e-3; evaluates **both** tasks and writes one JSON with a `test_segmentation`
  and a `test_classification` block.
- `../common/` — shared cleaning + grouped **leakage-free 3-class split**
  (`make_cls_split`) — identical to DenseNet/EfficientNet, so the **classification**
  number is directly comparable. Segmentation is scored on the benign+malignant subset
  of that same leakage-free test set.

## Faithful to the paper
Res-UNet + GroupNorm; OCA/OCR object-contextual segmentation head (K=2); GAP→2-layer-
MLP 3-class head classifying from **backbone** features (their ablation's best config);
combined loss `0.4·CE(soft)+CE(final)+CE(cls)`; Adam lr 1e-3; 2-channel grayscale+edge
input; augmentation (rotate ±45°, flip, brightness/contrast).

## Deviations (documented, deliberate)
| Paper | Ours | Why |
|---|---|---|
| 128×128 input | 256×256 (4-level Res-UNet) | match the rest of the project's pipeline |
| classification head = `Flatten → MLP` on full-res maps | **GAP → MLP** | Flatten on 256² maps is memory-infeasible; GAP is the standard fix |
| 400 epochs, 5-fold CV | 150 epochs, single leakage-free split | compute budget + our honest split protocol |
| exact OCR conv internals unspecified | standard OCR 1×1-conv/GN/ReLU transforms | paper does not specify them |
| seg Dice includes normal (empty-mask) images | seg scored on benign+malignant only | consistent with the other seg baselines |

Run: `python Models/MTL_OCA_Lu2025/train.py` → writes `Models/results/MTL_OCA_Lu2025.json`.
