# -*- coding: utf-8 -*-
"""Append the Models / Reproduction section to COMP9444_EDA.ipynb.
Only Models + Reproduction (no Results, no Discussion). Metrics stay in
Models/results/*.json for a later analysis stage. Each model shows its training curve."""
import os, nbformat as nbf
from nbformat.v4 import new_markdown_cell, new_code_cell

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NB   = os.path.join(PROJ, "COMP9444_EDA.ipynb")
nb = nbf.read(NB, as_version=4)

MARK = "## 4. Models"                       # truncate everything from the Models section on
cells = []
for c in nb.cells:
    if c.cell_type == "markdown" and c.source.lstrip().startswith(MARK):
        break
    cells.append(c)

def md(t):  cells.append(new_markdown_cell(t))
def code(t): cells.append(new_code_cell(t))

# ============================ 4. MODELS ============================ #
md(r"""## 4. Models

We reproduce **three deep-learning models** drawn from the project's relevant papers, all for the
breast-ultrasound lesion task:

| # | Paper | Model | What it is |
|---|---|---|---|
| **[03]** | Ronneberger et al., *U-Net*, MICCAI 2015 | **U-Net** (from scratch) | encoder–decoder + skip connections, pixel-wise segmentation |
| **[02]** | Yap et al., IEEE JBHI 2018 | **Patch-based LeNet** | 28×28 patch lesion/non-lesion classifier + sliding window (detection) |
| **[02]** | Yap et al., IEEE JBHI 2018 (*their best*) | **FCN-AlexNet** (transfer learning) | ImageNet-pretrained AlexNet made fully convolutional |

> Yap [02] proposes three methods (Patch-LeNet, U-Net, FCN-AlexNet); its **U-Net is the same
> architecture as [03]** (Yap cites Ronneberger), so across both papers there are **3 unique
> models**, all reproduced here.

All model code is under [`Models/`](Models/); this notebook only **calls** it.
```
Models/
├── common/   data.py (cleaning, split, datasets)  metrics.py  runner.py
├── UNet_Ronneberger2015/       model.py  train.py  README.md
├── Yap2018_PatchLeNet/         model.py  train.py  README.md
└── Yap2018_FCN_AlexNet/        model.py  train.py  README.md
```

*Evaluation metrics for every model are computed during training and saved to
`Models/results/*.json`; we analyse and compare them in a later results stage (not in this section).*""")

# ---------------- 4.1 Reproduction ----------------
md(r"""### 4.1 Reproduction

Per the marking guideline *"if building on previous work, identify the source and clearly
delineate which parts are your own work"*:

| Component | Source (previous work) | **Our own work** |
|---|---|---|
| U-Net architecture | Ronneberger et al. 2015 (arXiv:1505.04597) | wrote `UNet_Ronneberger2015/model.py` from scratch |
| Patch-LeNet architecture | Yap et al. 2018, Fig. 3 (no official code) | wrote `Yap2018_PatchLeNet/model.py` from scratch (+BatchNorm) |
| FCN-AlexNet | Yap et al. 2018 §IV-A-3; FCN (Long et al. 2015); AlexNet ImageNet weights (torchvision) | wrote the FCN head + fine-tuning; used the pretrained AlexNet backbone |
| Data cleaning, grouped leakage-free split, letterbox, augmentation | motivated by our EDA (§3) | **entirely ours** — `common/data.py` |
| Training loops, losses, sliding-window inference, ROI selection, metrics | — | **entirely ours** — `train.py`, `common/metrics.py` |

Deviations from the original papers are documented in each model's `README.md`.

**Shared preprocessing & split** (`Models/common/data.py`, from the EDA §3): drop the exact
cross-class duplicate (§3.6) · union multi-masks (§3.5) · letterbox to 256×256 + normalise (§3.4)
· **grouped, stratified split** so no near-duplicate crosses train/val/test (§3.6). Segmentation
uses the 645 benign+malignant images.""")
code(r"""import os, sys, json, subprocess
sys.path.insert(0, os.path.abspath("."))
from Models.common import data as D
import torch
from IPython.display import Image, display

split_df = D.make_split()
tbl = split_df.groupby(["split", "cls"]).size().unstack(fill_value=0); tbl["total"] = tbl.sum(1)
print("clusters spanning >1 split (must be 0):",
      int((split_df.groupby("cluster")["split"].nunique() > 1).sum()))
tbl""")

md(r"""#### Training / reproducing
Each script trains its model and writes metrics to `Models/results/<name>.json` (+ a training-curve
figure and a checkpoint). The helper below **loads** a saved result if present, else **trains once**.
To retrain, delete the JSON (or run e.g. `python Models/UNet_Ronneberger2015/train.py`).""")
code(r"""def get_results(name, script):
    path = os.path.join("Models", "results", f"{name}.json")
    if not os.path.exists(path):
        print(f"[train] {name} (runs the GPU training once)..."); subprocess.run([sys.executable, script], check=True)
    return json.load(open(path, encoding="utf-8"))

unet = get_results("UNet_Ronneberger2015",    "Models/UNet_Ronneberger2015/train.py")
yap  = get_results("Yap2018_PatchBasedLeNet",  "Models/Yap2018_PatchLeNet/train.py")
fcn  = get_results("Yap2018_FCN_AlexNet",      "Models/Yap2018_FCN_AlexNet/train.py")
print("Trained/loaded. Per-model metrics are saved in Models/results/*.json for the later results stage.")""")

# ---- (a) U-Net ----
md(r"""#### (a) U-Net — Ronneberger et al. 2015 [03]
Encoder–decoder with skip connections (`UNet_Ronneberger2015/model.py`), trained **from scratch**
on the 445 training images with a Dice+BCE loss. Deviations from the 2015 paper (padded convs,
BatchNorm, Dice+BCE) are documented in its `README.md`. Training curve below.""")
code(r"""from Models.UNet_Ronneberger2015.model import UNet
_m = UNet(1, 1, base=64)
print(f"U-Net  |  {sum(p.numel() for p in _m.parameters())/1e6:.1f} M params  |  trained {unet['train_minutes']} min"); del _m
display(Image("Models/results/figures/UNet_Ronneberger2015_curve.png"))""")

# ---- (b) Patch-LeNet ----
md(r"""#### (b) Patch-based LeNet — Yap et al. 2018 [02]
A LeNet classifies 28×28 patches as lesion / non-lesion; at test a sliding window builds a lesion
map that we threshold and reduce to the most-confident region(s) (Yap's ROI selection). It is a
**detection** method. Our additions (documented in its `README.md`): BatchNorm, hard-negative patch
sampling, top-strip masking, and a validation-tuned decision threshold. Training curve below.""")
code(r"""from Models.Yap2018_PatchLeNet.model import PatchLeNet
_m = PatchLeNet()
print(f"Patch-LeNet  |  {sum(p.numel() for p in _m.parameters())/1e6:.3f} M params  |  trained {yap['train_minutes']} min"); del _m
display(Image("Models/results/figures/Yap2018_PatchBasedLeNet_curve.png"))""")

# ---- (c) FCN-AlexNet ----
md(r"""#### (c) FCN-AlexNet (transfer learning) — Yap et al. 2018 [02] — *their best method*
An **ImageNet-pretrained AlexNet** made fully convolutional and fine-tuned for lesion delineation
(Yap §IV-A-3). Faithful settings from the paper: SGD lr 0.001, 60 epochs, dropout 0.33. The
pretrained backbone is the "transfer learning"; the FCN head and fine-tuning are ours (see
`README.md`). Training curve below.""")
code(r"""from Models.Yap2018_FCN_AlexNet.model import FCNAlexNet
_m = FCNAlexNet(pretrained=False)
print(f"FCN-AlexNet  |  {sum(p.numel() for p in _m.parameters())/1e6:.1f} M params  |  trained {fcn['train_minutes']} min"); del _m
display(Image("Models/results/figures/Yap2018_FCN_AlexNet_curve.png"))""")

nb.cells = cells
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nbf.write(nb, open(NB, "w", encoding="utf-8"))
print("wrote", NB, "| total cells:", len(cells))
