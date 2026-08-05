# -*- coding: utf-8 -*-
"""Models/common/runner.py -- shared helpers: seeding, results JSON, qualitative figures."""
import os, json, random
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
PROJ  = os.path.dirname(os.path.dirname(_HERE))
RESULTS_DIR = os.path.join(os.path.dirname(_HERE), "results")
FIG_DIR     = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def save_results(name, obj):
    """Write Models/results/<name>.json  (name = paper+model, human-readable)."""
    path = os.path.join(RESULTS_DIR, f"{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    print(f"[runner] wrote {path}")
    return path


def plot_curve(history, name, keys=("train_loss", "val_dice")):
    fig, ax = plt.subplots(1, len(keys), figsize=(5*len(keys), 4))
    if len(keys) == 1:
        ax = [ax]
    for a, k in zip(ax, keys):
        if k in history:
            a.plot(history[k]); a.set_title(k); a.set_xlabel("epoch"); a.grid(alpha=.3)
    fig.suptitle(name); fig.tight_layout()
    p = os.path.join(FIG_DIR, f"{name}_curve.png")
    fig.savefig(p, dpi=120); plt.close(fig)
    print(f"[runner] wrote {p}")
    return p


def plot_confusion(cm, class_names, name):
    """Save a labelled confusion-matrix heatmap for a classification model."""
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(4.6, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names))); ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names))); ax.set_yticklabels(class_names)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(f"{name} - confusion matrix")
    thr = cm.max() / 2 if cm.max() else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thr else "black")
    fig.colorbar(im, fraction=0.046, pad=0.04); fig.tight_layout()
    p = os.path.join(FIG_DIR, f"{name}_confusion.png")
    fig.savefig(p, dpi=120); plt.close(fig)
    print(f"[runner] wrote {p}")
    return p


def save_seg_examples(examples, name, n=6):
    """examples: list of (gray256[-1..1 or 0..255], gt_mask, pred_mask). Save a grid."""
    n = min(n, len(examples))
    fig, ax = plt.subplots(n, 3, figsize=(9, 3*n))
    if n == 1:
        ax = ax[None, :]
    for i in range(n):
        g, gt, pr = examples[i]
        g = np.asarray(g, dtype=np.float32)
        if g.min() < 0:                       # undo [-1,1] normalisation for display
            g = (g*0.5 + 0.5)
            g = (g*255).clip(0, 255)
        ax[i, 0].imshow(g, cmap="gray");            ax[i, 0].set_ylabel(f"test {i}")
        ax[i, 1].imshow(gt, cmap="gray")
        ax[i, 2].imshow(g, cmap="gray")
        ax[i, 2].imshow(np.ma.masked_where(pr == 0, pr), cmap="autumn", alpha=.6)
        if i == 0:
            ax[i, 0].set_title("image"); ax[i, 1].set_title("ground truth"); ax[i, 2].set_title("prediction")
        for a in ax[i]:
            a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"{name} - qualitative results (test set)"); fig.tight_layout()
    p = os.path.join(FIG_DIR, f"{name}_examples.png")
    fig.savefig(p, dpi=120); plt.close(fig)
    print(f"[runner] wrote {p}")
    return p
