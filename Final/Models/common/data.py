# -*- coding: utf-8 -*-
"""
Models/common/data.py
Data cleaning + preprocessing + datasets for the BUSI baselines, driven by the EDA.

Cleaning steps (all motivated by the EDA in COMP9444_EDA.ipynb, section 3.6):
  1. drop the exact cross-class duplicate  (benign(433) == malignant(145))
  2. merge multiple masks per image into a single binary mask (pixel-wise union)
  3. build near-duplicate CLUSTERS (pHash) and make a GROUPED, stratified
     train/val/test split so no near-duplicate crosses the split boundary
     (prevents the leakage quantified in EDA 3.6 / fig10).
Preprocessing (EDA 3.4/3.5): letterbox-resize to 256x256 (keep aspect, pad),
grayscale, normalise to [-1,1]. Augmentation (train only): flip / affine / brightness.

Segmentation uses the 645 benign+malignant images (they have real lesions).
Classification uses ALL THREE classes (normal/benign/malignant) -- see make_cls_split.
This module is imported by every model training script so they share one clean split.
"""
import os, sys, hashlib
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A

# --- make the project root importable so we can reuse utils/eda_utils.py --------
_HERE = os.path.dirname(os.path.abspath(__file__))
PROJ  = os.path.dirname(os.path.dirname(_HERE))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)
from utils import eda_utils as eda   # noqa: E402

DATA_DIR   = os.path.join(PROJ, "datasets", "Breast-Cancer-Ultrasound-Images-Dataset",
                          "Dataset_BUSI_with_GT")
CACHE_META = os.path.join(PROJ, "EDA", "artifacts", "nb_metadata.csv")
SPLIT_CSV     = os.path.join(_HERE, "split.csv")       # segmentation (benign+malignant)
CLS_SPLIT_CSV = os.path.join(_HERE, "split_cls.csv")   # classification (all 3 classes)
IMG_SIZE   = 256
PATCH      = 28          # Yap patch size
SEED       = 42
CLASS_NAMES = ("normal", "benign", "malignant")        # 3-class order for classification
LABEL2ID    = {c: i for i, c in enumerate(CLASS_NAMES)}


# ----------------------------------------------------------------------------- #
# 1. cleaned, grouped, stratified split
# ----------------------------------------------------------------------------- #
def _grouped_stratified_split(df, cluster_col, class_col, fracs=(0.70, 0.15, 0.15), seed=SEED):
    """Assign whole clusters to train/val/test, stratified by class. No cluster is split."""
    rng = np.random.RandomState(seed)
    # one representative class per cluster (majority)
    clu_class = (df.groupby(cluster_col)[class_col]
                   .agg(lambda s: s.value_counts().index[0]))
    split_of = {}
    for cls in sorted(df[class_col].unique()):
        clus = clu_class[clu_class == cls].index.tolist()
        rng.shuffle(clus)
        n = len(clus); n_tr = int(round(fracs[0]*n)); n_va = int(round(fracs[1]*n))
        for i, c in enumerate(clus):
            split_of[c] = "train" if i < n_tr else ("val" if i < n_tr+n_va else "test")
    return df[cluster_col].map(split_of)


def _clean_and_cluster():
    """Shared cleaning used by BOTH split builders: build metadata, drop the exact
    cross-class duplicate (label conflict), and tag every image with a near-duplicate
    CLUSTER id (pHash) so the split can be made leakage-free. Returns the cleaned
    dataframe with a 'cluster' column but no 'split' yet."""
    df = eda.build_metadata(DATA_DIR, cache_path=CACHE_META, recompute=False, verbose=False)

    # --- clean 1: drop exact cross-class duplicate group (label conflict) --------
    dup = df.groupby("img_md5").filter(lambda g: g.cls.nunique() > 1)
    df = df[~df.filename.isin(dup.filename)].copy()

    # --- clusters from near-duplicates (pHash) for a leakage-free split ----------
    near, clusters = eda.find_near_duplicates(df, thresh=5)
    cid = {}
    for i, (_, members) in enumerate(clusters.items()):
        for m in members:
            cid[m] = f"c{i}"
    # singletons get their own cluster id
    df["cluster"] = [cid.get(fn, f"s{k}") for k, fn in enumerate(df.filename)]
    return df


def _report_split(df, path, verbose):
    if verbose:
        print(f"[data] built split -> {path}")
        print(df.groupby(["split", "cls"]).size().unstack(fill_value=0))
        # leakage sanity check: every cluster lives in exactly one split
        bad = (df.groupby("cluster")["split"].nunique() > 1).sum()
        print(f"[data] clusters spanning >1 split (should be 0): {bad}")


def make_split(seg_only=True, seed=SEED, force=False, verbose=True):
    """SEGMENTATION split: benign+malignant only (images that actually contain a
    lesion), grouped + stratified, cached to split.csv."""
    if os.path.exists(SPLIT_CSV) and not force:
        df = pd.read_csv(SPLIT_CSV)
        if verbose:
            print(f"[data] loaded cached split: {SPLIT_CSV}")
        return df

    df = _clean_and_cluster()
    if seg_only:
        df = df[df.cls.isin(["benign", "malignant"])].copy()

    df["split"] = _grouped_stratified_split(df, "cluster", "cls", seed=seed)
    keep = ["filename", "cls", "path", "stem", "width", "height",
            "n_masks", "area_ratio", "cluster", "split"]
    df = df[keep].reset_index(drop=True)
    df.to_csv(SPLIT_CSV, index=False, encoding="utf-8")
    _report_split(df, SPLIT_CSV, verbose)
    return df


def make_cls_split(seed=SEED, force=False, verbose=True):
    """CLASSIFICATION split: ALL THREE classes (normal/benign/malignant), grouped by
    near-duplicate cluster + stratified by class, cached to split_cls.csv. Adds an
    integer 'label' column (normal=0, benign=1, malignant=2). Identical cleaning and
    leakage protection as make_split; the only difference is that `normal` images are
    KEPT (they have no lesion, so they exist for classification but not segmentation)."""
    if os.path.exists(CLS_SPLIT_CSV) and not force:
        df = pd.read_csv(CLS_SPLIT_CSV)
        if verbose:
            print(f"[data] loaded cached classification split: {CLS_SPLIT_CSV}")
        return df

    df = _clean_and_cluster()
    df["split"] = _grouped_stratified_split(df, "cluster", "cls", seed=seed)
    df["label"] = df.cls.map(LABEL2ID).astype(int)
    keep = ["filename", "cls", "label", "path", "stem", "width", "height",
            "n_masks", "area_ratio", "cluster", "split"]
    df = df[keep].reset_index(drop=True)
    df.to_csv(CLS_SPLIT_CSV, index=False, encoding="utf-8")
    _report_split(df, CLS_SPLIT_CSV, verbose)
    return df


# ----------------------------------------------------------------------------- #
# 2. letterbox (resize keeping aspect + pad)  -> image, mask, valid-region
# ----------------------------------------------------------------------------- #
def load_gray_and_mask(row):
    """Read one image (grayscale) and its union mask at original resolution."""
    g = cv2.imread(row["path"], cv2.IMREAD_GRAYSCALE)
    H, W = g.shape
    m = eda.load_union_mask(row["path"], (H, W))       # union of multi-masks, {0,1}
    return g, m


def letterbox(img, mask, size=IMG_SIZE):
    """Resize longest side to `size`, pad to a square. Returns img, mask, valid(0/1)."""
    h, w = img.shape[:2]
    s = size / max(h, w)
    nh, nw = int(round(h*s)), int(round(w*s))
    img_r  = cv2.resize(img,  (nw, nh), interpolation=cv2.INTER_AREA)
    mask_r = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
    canvas_i = np.zeros((size, size), np.uint8)
    canvas_m = np.zeros((size, size), np.uint8)
    valid    = np.zeros((size, size), np.uint8)
    y0, x0 = (size-nh)//2, (size-nw)//2
    canvas_i[y0:y0+nh, x0:x0+nw] = img_r
    canvas_m[y0:y0+nh, x0:x0+nw] = mask_r
    valid[y0:y0+nh, x0:x0+nw] = 1
    return canvas_i, canvas_m, valid


def _norm_to_tensor(gray_uint8):
    """256x256 uint8 grayscale -> (1,256,256) float tensor in [-1,1]."""
    x = gray_uint8.astype(np.float32) / 255.0
    x = (x - 0.5) / 0.5
    return torch.from_numpy(x).unsqueeze(0)


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], np.float32)

def _norm_to_tensor3(gray_uint8):
    """256x256 uint8 grayscale -> (3,256,256) ImageNet-normalised tensor
    (for pretrained backbones such as FCN-AlexNet; BUSI is pseudo-RGB so we
    replicate the single channel)."""
    x = gray_uint8.astype(np.float32) / 255.0
    x = np.stack([x, x, x], 0)                       # (3,H,W)
    x = (x - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]
    return torch.from_numpy(x)


# ----------------------------------------------------------------------------- #
# 3. Segmentation dataset (for U-Net [03])
# ----------------------------------------------------------------------------- #
def _train_aug():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Affine(scale=(0.9, 1.1), translate_percent=0.05, rotate=(-15, 15),
                 border_mode=cv2.BORDER_CONSTANT, p=0.7),
        A.RandomBrightnessContrast(0.2, 0.2, p=0.3),
    ])


class SegDataset(Dataset):
    """Letterboxed 256x256 grayscale image + binary lesion mask.
    Images are pre-loaded into memory once (fast epochs). Set rgb3=True to output
    3-channel ImageNet-normalised tensors for pretrained backbones (FCN-AlexNet)."""
    def __init__(self, df, split, augment=False, rgb3=False):
        rows = df[df.split == split].reset_index(drop=True)
        self.augment = augment
        self.rgb3 = rgb3
        self.aug = _train_aug() if augment else None
        self.cache = []                              # preload letterboxed (gray, mask)
        for _, row in rows.iterrows():
            g, m = load_gray_and_mask(row)
            g, m, _ = letterbox(g, m)
            self.cache.append((g, (m > 0).astype(np.uint8)))

    def __len__(self):
        return len(self.cache)

    def __getitem__(self, i):
        g, m = self.cache[i]
        if self.aug is not None:
            out = self.aug(image=g, mask=m)
            g, m = out["image"], out["mask"]
        x = _norm_to_tensor3(g) if self.rgb3 else _norm_to_tensor(g)
        y = torch.from_numpy((m > 0).astype(np.float32)).unsqueeze(0)
        return x, y


# ----------------------------------------------------------------------------- #
# 3b. Classification dataset (for ResNet [04] / DenseNet / EfficientNet ...)
# ----------------------------------------------------------------------------- #
def _letterbox_img(img, size=IMG_SIZE):
    """Letterbox an image only (no mask): resize longest side to `size`, pad square."""
    h, w = img.shape[:2]
    s = size / max(h, w)
    nh, nw = int(round(h*s)), int(round(w*s))
    img_r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size), np.uint8)
    y0, x0 = (size-nh)//2, (size-nw)//2
    canvas[y0:y0+nh, x0:x0+nw] = img_r
    return canvas


def _train_aug_cls():
    """Image-only augmentation for classification (no mask to keep in sync)."""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Affine(scale=(0.9, 1.1), translate_percent=0.05, rotate=(-15, 15),
                 border_mode=cv2.BORDER_CONSTANT, p=0.7),
        A.RandomBrightnessContrast(0.2, 0.2, p=0.3),
    ])


class ClsDataset(Dataset):
    """Letterboxed 256x256 grayscale image + a 3-class label (normal/benign/malignant).
    Images are pre-loaded into memory once (fast epochs). rgb3=True (the default) outputs
    3-channel ImageNet-normalised tensors for ImageNet-pretrained backbones such as
    ResNet [04]; set rgb3=False for a single [-1,1] channel (from-scratch models)."""
    def __init__(self, df, split, augment=False, rgb3=True):
        rows = df[df.split == split].reset_index(drop=True)
        self.aug = _train_aug_cls() if augment else None
        self.rgb3 = rgb3
        self.cache = []                              # preload (letterboxed gray, label)
        for _, row in rows.iterrows():
            g = cv2.imread(row["path"], cv2.IMREAD_GRAYSCALE)
            self.cache.append((_letterbox_img(g), int(row["label"])))

    def __len__(self):
        return len(self.cache)

    def __getitem__(self, i):
        g, y = self.cache[i]
        if self.aug is not None:
            g = self.aug(image=g)["image"]
        x = _norm_to_tensor3(g) if self.rgb3 else _norm_to_tensor(g)
        return x, torch.tensor(y, dtype=torch.long)


def class_weights(df, split="train"):
    """Inverse-frequency class weights over `split` (for imbalanced CrossEntropy).
    normal/benign/malignant are imbalanced, so we up-weight rarer classes."""
    counts = (df[df.split == split].groupby("label").size()
              .reindex(range(len(CLASS_NAMES)), fill_value=0))
    w = counts.sum() / (len(CLASS_NAMES) * counts.clip(lower=1))
    return torch.tensor(w.values, dtype=torch.float32)


# ----------------------------------------------------------------------------- #
# 4. Patch dataset (for Yap Patch-LeNet [02])
# ----------------------------------------------------------------------------- #
class PatchDataset(Dataset):
    """28x28 grayscale patches labelled lesion(1) / non-lesion(0).
    Positives: patch centred inside the lesion. Negatives: patch fully outside the
    lesion and inside the valid (non-padded) region. Sampled per epoch."""
    def __init__(self, df, split, per_image=60, augment=False, seed=SEED):
        self.items = []                       # (letterboxed gray, mask, valid)
        for _, row in df[df.split == split].iterrows():
            g, m = load_gray_and_mask(row)
            self.items.append(letterbox(g, m))
        self.per_image = per_image
        self.augment = augment
        self.rng = np.random.RandomState(seed)
        self.half = PATCH // 2
        self._build()

    def _sample_centres(self, g, mask, valid, k, mode):
        """mode: 'pos' (in lesion) | 'neg' (any background) | 'neg_hard'
        (BRIGHT background or the TOP band -- the main false-positive sources)."""
        h = self.half
        region = np.zeros_like(valid)
        region[h:IMG_SIZE-h, h:IMG_SIZE-h] = 1                # keep full patch in-bounds
        bg = (mask == 0) & (valid > 0) & (region > 0)
        if mode == "pos":
            ys, xs = np.where((mask > 0) & (region > 0))
        elif mode == "neg":
            ys, xs = np.where(bg)
        else:  # neg_hard: bright tissue / probe line / top rows
            thr = np.percentile(g[valid > 0], 75) if (valid > 0).any() else 128
            top = np.zeros_like(valid); top[:int(0.20*IMG_SIZE)] = 1
            ys, xs = np.where(bg & ((g >= thr) | (top > 0)))
            if len(ys) == 0:
                ys, xs = np.where(bg)
        if len(ys) == 0:
            return []
        idx = self.rng.randint(0, len(ys), size=k)
        return list(zip(ys[idx], xs[idx]))

    def _build(self):
        self.patches, self.labels = [], []
        q = self.per_image // 4
        for g, m, v in self.items:
            plan = [(1, "pos", 2*q), (0, "neg", q), (0, "neg_hard", q)]   # 50% pos, 25% easy-neg, 25% hard-neg
            for lbl, mode, k in plan:
                for (cy, cx) in self._sample_centres(g, m, v, k, mode):
                    p = g[cy-self.half:cy+self.half, cx-self.half:cx+self.half]
                    if p.shape == (PATCH, PATCH):
                        self.patches.append(p); self.labels.append(lbl)

    def resample(self):
        """Draw a fresh set of patches (call once per epoch for variety)."""
        self._build()

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, i):
        p = self.patches[i].copy()
        if self.augment and self.rng.rand() < 0.5:
            p = np.fliplr(p).copy()
        x = _norm_to_tensor(p)
        return x, torch.tensor(self.labels[i], dtype=torch.long)


def iter_test_images(df, split="test"):
    """Yield (row, gray256, mask256, valid256) for sliding-window evaluation."""
    for _, row in df[df.split == split].iterrows():
        g, m = load_gray_and_mask(row)
        gi, mi, vi = letterbox(g, m)
        yield row, gi, mi, vi


if __name__ == "__main__":
    d = make_split(force=True)
    print("\nTotal:", len(d))
    print(d.groupby(["split", "cls"]).size().unstack(fill_value=0))
