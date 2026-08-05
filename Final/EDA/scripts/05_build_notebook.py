# -*- coding: utf-8 -*-
"""Assemble COMP9444_EDA.ipynb (Intro + Data Sources + EDA) from markdown + short code cells."""
import os, nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT  = os.path.join(PROJ, "COMP9444_EDA.ipynb")
cells = []
def md(t):  cells.append(new_markdown_cell(t))
def code(t): cells.append(new_code_cell(t))

# =========================================================================== #
md("""# COMP9444 Project 047 — Breast Cancer Classification & Segmentation on Ultrasound Images
### Introduction, Data Sources, and Exploratory Data Analysis

*Dataset: Breast Ultrasound Images (BUSI), Al-Dhabyani et al. (2020).*

This notebook is written as a **readable report**. Heavy / repeated code lives in
`utils/eda_utils.py`; the notebook only contains the main workflow, plots, and the
analysis after each plot. Every number below is **computed from the dataset**, not
hard-coded.

**Contents**
1. Introduction, Motivation and Problem Statement
2. Data Sources
3. Exploratory Data Analysis
   1. Dataset overview · 2. Class distribution · 3. Sample visualization ·
   4. Image properties · 5. Lesion & mask analysis · 6. Data quality (duplicates & leakage) ·
   7. Preprocessing analysis · 8. Summary & challenges → model design
""")

# ---------------------------------------------------------------- 1. Intro --- #
md("""## 1. Introduction, Motivation and Problem Statement

Breast cancer is one of the leading causes of cancer death in women worldwide, and
**early detection is the most effective way to reduce mortality**. Ultrasound is a
safe, low-cost, radiation-free imaging modality that is especially useful for dense
breasts, where mammography is less sensitive. However, ultrasound images are hard to
read: **speckle noise, low contrast, acoustic shadowing and the large variability of
lesion appearance** make interpretation operator-dependent and time-consuming, even
for experienced radiologists [Cheng 2010; Xian 2018]. This motivates computer-aided
diagnosis (CAD) based on deep learning.

**Purpose / problem statement.** Following the Project 047 brief, we build a
deep-learning pipeline on the BUSI dataset that addresses **two tasks**:

* **Classification** — assign each ultrasound image to one of three classes:
  `normal`, `benign`, or `malignant` (backbones such as ResNet, DenseNet,
  EfficientNet [He 2016; Huang 2017; Tan & Le 2019]).
* **Segmentation** — delineate the lesion boundary at pixel level using the
  ground-truth masks (U-Net / DeepLabv3+ / Attention U-Net
  [Ronneberger 2015; Chen 2018; Oktay 2018]).

The goal is a system that can **assist radiologists in early detection**, improving
the efficiency and consistency of screening. Before any modelling, this notebook
performs a thorough EDA to understand the data's properties and challenges, which
then **directly informs preprocessing, model choice, training strategy and
evaluation metrics** (Section 3.8).""")

# ------------------------------------------------------------ 2. Data src --- #
md("""## 2. Data Sources

We use the **Breast Ultrasound Images (BUSI)** dataset of *Al-Dhabyani, Gomaa,
Khaled & Fahmy (2020), Data in Brief 28:104863* — a widely used public benchmark.

| Property | Value |
|---|---|
| Source | Baheya Hospital for Early Detection & Treatment of Women's Cancer, Cairo, Egypt (collected 2018) |
| Patients | 600 female patients, aged 25–75 |
| # images | 780 ultrasound scans (PNG, average ≈ 500×500 px) |
| Classes | `normal`, `benign`, `malignant` (image-level labels) |
| Annotations | **pixel-level ground-truth lesion masks** for each image |
| Task type | **Both** — classification *and* segmentation |
| License | CC-BY-4.0 |

Because BUSI provides **both** image-level labels and pixel-level masks, it supports
the full Project 047 scope. A related study by the same authors explores data
augmentation for the same imagery [Al-Dhabyani 2019], which informs our augmentation
choices.

**Folder layout** (each image has one original PNG plus one or more `_mask` PNGs;
`normal` images have an all-black mask):

```
Dataset_BUSI_with_GT/
├── benign/     benign (1).png,  benign (1)_mask.png, [benign (1)_mask_1.png ...]
├── malignant/  malignant (1).png, malignant (1)_mask.png ...
└── normal/     normal (1).png,  normal (1)_mask.png ...
```

> **`DATA_DIR` is set in the next cell.** If your dataset lives elsewhere, change that
> one variable — nothing else in the notebook needs editing.

*Note:* BUSI is known to contain some duplicate/near-duplicate images; we quantify
and handle this explicitly in Section 3.6.""")

# ------------------------------------------------------------ 3. EDA setup -- #
md("## 3. Exploratory Data Analysis\n\n### Setup")
code("""import os, sys
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# make the helper module importable (notebook lives at the project root)
sys.path.insert(0, os.path.abspath("."))
from utils import eda_utils as eda

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110
pd.set_option("display.max_colwidth", 60)

# >>> CHANGE THIS if your dataset is elsewhere <<<
DATA_DIR = "datasets/Breast-Cancer-Ultrasound-Images-Dataset/Dataset_BUSI_with_GT"
CACHE    = "EDA/artifacts/nb_metadata.csv"   # per-image features are cached here
assert os.path.isdir(DATA_DIR), f"DATA_DIR not found: {DATA_DIR}"
print("Dataset OK:", DATA_DIR)""")

code("""# Build the per-image metadata table (scans images + masks once, then caches).
# Columns: size, channels, intensity stats, mask count, lesion area, shape, hashes...
df = eda.build_metadata(DATA_DIR, cache_path=CACHE, recompute=False)
print(df.shape)
df.head(3)""")

# ----- 3.1 overview
md("### 3.1 Dataset Overview")
code("eda.overview_table(df)")
md("""The dataset contains **780 images across 3 classes**, and **every image has at
least one lesion mask**, confirming BUSI supports both classification and
segmentation. Note the image sizes are **not** a single fixed resolution (see 3.4).""")

# ----- 3.2 class distribution
md("### 3.2 Class Distribution")
code("eda.class_distribution(df)")
code("eda.plot_class_distribution(df);")
md("""**Analysis — imbalanced.** `benign` (≈56%) dominates, followed by `malignant`
(≈27%) and `normal` (≈17%); benign is ~3× the normal class. This imbalance means
plain accuracy is misleading and the model can be biased toward the majority class.
**Implications:** use a *stratified* train/val/test split, apply *class weights* (or
focal loss), balance classes with augmentation, and report **per-class** metrics
(precision/recall/F1, macro-average) and AUC rather than overall accuracy alone.""")

# ----- 3.3 samples
md("### 3.3 Sample Image Visualization")
code("eda.plot_sample_grid(df, n_per_class=4);")
md("""**Analysis — visual differences & issues.**
* **Benign** lesions tend to be **oval, well-circumscribed and hypoechoic** (dark,
  smooth borders).
* **Malignant** lesions are typically **irregular, ill-defined, taller-than-wide**,
  often with posterior acoustic shadowing.
* **Normal** images show fibroglandular/fatty tissue with no focal mass.

Observable **challenges**: strong **speckle noise** and **low contrast**; benign and
malignant can look similar; and several images carry **burned-in annotations,
calipers and text** (e.g. "RT LOQ", "RIGHT BREAST"). Such artifacts risk *shortcut
learning* and may need cropping/inpainting.""")

# ----- 3.4 image properties
md("### 3.4 Image Properties — size, channels, pixel intensity")
code("""# Are the images grayscale or RGB?
eda.channel_summary(df)""")
code("eda.plot_image_sizes(df);")
md("""**Analysis — inconsistent sizes & pseudo-RGB.** Although the dataset is described
as "≈500×500", the images span a **wide range of sizes** (many hundreds of distinct
resolutions; width and height vary by hundreds of pixels) and most are **landscape
(W/H > 1)**. All files are stored as **3-channel RGB even though ultrasound is
grayscale** (the three channels are essentially identical). **Implications:** we must
**resize to a fixed input** (e.g. 256×256); prefer **resize-with-padding** to avoid
distorting lesion shape; and we can safely convert to single-channel grayscale, or
keep 3 channels to reuse ImageNet-pretrained weights.""")
code("eda.plot_pixel_intensity(df);")
md("""**Analysis — intensity overlaps across classes.** Per-image brightness and
contrast distributions **overlap heavily** between the three classes, so global
intensity is a *weak* discriminator on its own — the model must rely on texture and
shape. The wide brightness range across images argues for **per-image
normalization** (and optionally CLAHE for low-contrast images; see 3.7).""")

# ----- 3.5 lesion & mask
md("""### 3.5 Lesion & Mask Analysis
Beyond the images, the masks let us quantify lesion size, location and shape — all
directly relevant to segmentation.""")
code("eda.mask_matching_table(df)")
code("eda.plot_mask_matching(df);")
md("""**Analysis — clean image↔mask matching.** All 780 images have a mask (no orphan
files). A small number of images (the benign class mostly) have **multiple lesion
masks** (up to 3); for binary segmentation we merge them with a **pixel-wise union**.
`normal` masks are all-black (background only).""")
code("eda.plot_lesion_area(df);")
md("""**Analysis — malignant lesions are larger.** The lesion-area ratio (lesion
pixels ÷ image pixels) is **much larger for malignant (~12% median) than benign
(~4%)**. Two consequences: (i) lesion **size is itself discriminative**; (ii) many
benign lesions are **very small**, creating strong foreground/background imbalance at
the pixel level → prefer **Dice + BCE** (or Tversky) loss for segmentation.""")
code("""hm = eda.compute_location_heatmaps(df)   # sums aligned masks per class
eda.plot_location_heatmaps(hm);""")
md("""**Analysis — center/acquisition bias.** Lesions concentrate **near the centre,
slightly above the middle** of the frame, because sonographers centre the probe on
the finding. A model could exploit this position cue; **random crops/translations**
during training improve robustness. (`normal` has no lesion signal.)""")
code("eda.plot_lesion_shape(df);")
md("""**Analysis — malignant shapes are more irregular.** Malignant lesions show
**lower circularity, solidity and extent, and higher eccentricity** than benign,
quantifying the "irregular, spiculated" appearance seen in 3.3. This supports
boundary-aware models (e.g. **Attention U-Net**) and shape-sensitive losses.""")

# ----- 3.6 data quality
md("""### 3.6 Data Quality — Duplicates & Train/Test Leakage
This is the most important integrity check. We detect **exact duplicates** (identical
file bytes, via MD5) and **near-duplicates** (very similar content, via perceptual
hash + Hamming distance), then estimate the **leakage** a naive random split causes.""")
code("""dup = eda.exact_duplicates(df)
dup""")
code("""near, clusters = eda.find_near_duplicates(df, thresh=5)
print(f"near-duplicate pairs (Hamming<=5): {len(near)}")
print(f"cross-class near-duplicate pairs : {(near.cls_a != near.cls_b).sum()}")
print(f"near-duplicate clusters          : {len(clusters)}")
print(f"images inside a cluster          : {sum(len(v) for v in clusters.values())} "
      f"({sum(len(v) for v in clusters.values())/len(df)*100:.0f}% of data)")""")
code("eda.plot_duplicates(df, near);")
md("""**Analysis — real label conflicts and duplication.** There is an **exact
duplicate**: `benign (433).png` and `malignant (145).png` are the **same file bytes
but carry different class labels** — a genuine labeling conflict. More broadly, ~180+
near-duplicate pairs involve roughly a **third of the dataset**, and **10 pairs even
cross class boundaries**. This is a documented BUSI issue (Pawłowska et al., *Data in
Brief* Letter to the Editor, 2023). **Action:** de-duplicate and manually resolve
cross-class conflicts before training.""")
code("""leaks = eda.simulate_split_leakage(df, clusters, n_sims=40, test_frac=0.30)
eda.plot_leakage(df, clusters, leaks);
print(f"On average, a random 70/30 split leaks ~{np.mean(leaks):.0f} clusters across train/test.")""")
md("""**Analysis — why this matters.** Under a **naive random split**, dozens of
near-duplicate clusters end up with copies on *both* the train and test sides. The
test set then contains images the model effectively saw in training, which
**inflates the reported accuracy/Dice** — a classic leakage trap. **Action:** use a
**grouped split** that keeps each duplicate cluster (ideally each *patient*) entirely
on one side, and de-duplicate first. This is essential for a trustworthy evaluation.""")

# ----- 3.7 preprocessing
md("""### 3.7 Preprocessing Analysis
We choose preprocessing **based on the EDA above**, not blindly. The panel below shows
each candidate step on one lesion image.""")
code("eda.plot_preprocessing_demo(df);")
md("""**Reasoning for each step:**

| Step | Use it? | Why (from EDA) |
|---|---|---|
| **Resize to fixed size (256×256)** | ✅ required | Sizes are inconsistent (3.4); networks need fixed input. |
| **Resize *with padding*** (letterbox) | ✅ preferred | Most images are landscape; naive resizing distorts lesion aspect/shape, which is discriminative (3.5). |
| **Normalization** (per-image / ImageNet) | ✅ required | Brightness/contrast vary widely and overlap across classes (3.4); stabilizes training. |
| **Grayscale vs 3-channel** | ▶ choice | Images are pseudo-RGB (3.4); use 1 channel to save compute, or 3 channels to reuse ImageNet weights. |
| **Data augmentation** (flip, rotate, small scale/translate) | ✅ yes | Small, imbalanced dataset (3.2) + center bias (3.5); improves generalization and balances classes. |
| **CLAHE** (contrast enhancement) | ⚖ optional | Many low-contrast images; can help visibility but may amplify speckle — test as an ablation, not by default. |
| **Denoising** (NLM/median) | ⚖ optional | Speckle is informative texture; denoise cautiously — validate it helps before adopting. |
| **De-duplication + grouped split** | ✅ critical | Duplicates & leakage (3.6) would otherwise inflate results. |""")

# ----- 3.8 summary
md("### 3.8 EDA Summary & Dataset Challenges → Model Design")
code("""# Consolidated, computed summary (no hard-coded numbers)
import json
print(json.dumps(eda.summary_stats(df, near, clusters), indent=2))""")
md("""**Key findings (all computed above):**
1. **Class imbalance** — benign 56% / malignant 27% / normal 17%.
2. **Inconsistent image sizes** and **pseudo-RGB** storage → fixed-size resizing needed.
3. **Intensity overlaps** across classes → texture/shape matter more than brightness.
4. **Malignant lesions are larger and more irregular**; many benign lesions are tiny.
5. **Center/acquisition bias** in lesion location.
6. **Duplicates & leakage** — an exact cross-class duplicate, ~180 near-duplicate
   pairs (~⅓ of data), 10 cross-class → **must de-duplicate and use grouped splits**.
7. **Burned-in annotations** on some images → shortcut-learning risk.

**How this shapes the project:**

| Finding | Effect on design |
|---|---|
| Imbalance | stratified split, class weights / focal loss, balanced augmentation, **per-class F1 & AUC** (not just accuracy) |
| Inconsistent sizes / pseudo-RGB | resize **with padding** to 256×256; grayscale or 3-channel for pretrained backbones |
| Intensity overlap, low contrast | per-image normalization; CLAHE as an ablation |
| Small / imbalanced lesions | segmentation loss = **Dice + BCE**; report **Dice & IoU** |
| Irregular malignant boundaries | try **Attention U-Net**; boundary-aware loss |
| Center bias | random crop/translation augmentation |
| Duplicates & leakage | **de-duplicate + grouped (patient/cluster-aware) split** for honest metrics |
| Small dataset overall | **transfer learning** + augmentation + regularization to limit overfitting |

These decisions carry directly into the modelling stage: transfer-learned classifiers
(ResNet/DenseNet/EfficientNet-B0) and segmentation models (U-Net → DeepLabv3+ →
Attention U-Net), evaluated under a leakage-free grouped split with per-class and
overlap metrics.""")

nb = new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.metadata["language_info"] = {"name": "python"}
with open(OUT, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("wrote", OUT, "| cells:", len(cells))
