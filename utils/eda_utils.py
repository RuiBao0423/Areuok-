# -*- coding: utf-8 -*-
"""
eda_utils.py  —  Helper functions for the Breast Ultrasound (BUSI) EDA notebook.

All heavy / repeated code lives here so the notebook stays readable. The notebook
imports this module and calls the functions below. Nothing here invents numbers:
every statistic is computed directly from the image and mask files on disk.

Main entry point:
    df = build_metadata(DATA_DIR)          # one row per original image

Plotting helpers each return a matplotlib Figure (shown inline in the notebook):
    plot_class_distribution, plot_sample_grid, plot_image_sizes,
    plot_pixel_intensity, plot_mask_matching, plot_multimask_examples,
    plot_lesion_area, plot_location_heatmaps, plot_lesion_shape, plot_duplicates,
    plot_leakage, plot_preprocessing_demo
"""
import os, re, hashlib, collections
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from PIL import Image
import imagehash

# ----------------------------------------------------------------------------- #
# Constants
# ----------------------------------------------------------------------------- #
CLASSES = ["benign", "malignant", "normal"]
COL = {"benign": "#2ca02c", "malignant": "#d62728", "normal": "#7f7f7f"}
HM = 256  # resolution used to align masks for the location heatmap
_NAME_RE = re.compile(r"^(benign|malignant|normal) \((\d+)\)\.png$", re.IGNORECASE)


# ----------------------------------------------------------------------------- #
# Low-level helpers
# ----------------------------------------------------------------------------- #
def _md5(path):
    """MD5 of the raw bytes -> detects byte-identical (exact) duplicate files."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def mask_paths_for(image_path):
    """Return the list of mask files belonging to one original image.
    Supports multiple lesions: '<stem>_mask.png', '<stem>_mask_1.png', ..."""
    d = os.path.dirname(image_path)
    stem = os.path.basename(image_path)[:-4]  # drop '.png'
    return sorted(os.path.join(d, f) for f in os.listdir(d)
                  if f.startswith(stem + "_mask") and f.endswith(".png"))


def load_union_mask(image_path, shape_hw):
    """Binary union of all masks of an image, resized to (H, W). White(>127)=lesion."""
    H, W = shape_hw
    union = np.zeros((H, W), np.uint8)
    for mp in mask_paths_for(image_path):
        mm = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        if mm is None:
            continue
        if mm.shape != (H, W):
            mm = cv2.resize(mm, (W, H), interpolation=cv2.INTER_NEAREST)
        union |= (mm > 127).astype(np.uint8)
    return union


def _shape_features(union):
    """Morphological descriptors of the largest lesion contour."""
    out = dict(circularity=np.nan, solidity=np.nan, extent=np.nan,
               bbox_aspect=np.nan, eccentricity=np.nan)
    cnts, _ = cv2.findContours(union, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return out
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    peri = cv2.arcLength(c, True)
    if area <= 0 or peri <= 0:
        return out
    hull = cv2.convexHull(c); harea = cv2.contourArea(hull)
    bx, by, bw, bh = cv2.boundingRect(c)
    out["circularity"] = 4 * np.pi * area / (peri ** 2)      # 1.0 = perfect circle
    out["solidity"]    = area / harea if harea > 0 else np.nan
    out["extent"]      = area / (bw * bh) if bw * bh > 0 else np.nan
    out["bbox_aspect"] = bw / bh if bh > 0 else np.nan
    if len(c) >= 5:                                          # ellipse needs >=5 pts
        (_, _), (MA, ma), _ = cv2.fitEllipse(c)
        a_, b_ = max(MA, ma) / 2.0, min(MA, ma) / 2.0
        out["eccentricity"] = np.sqrt(1 - (b_ ** 2) / (a_ ** 2)) if a_ > 0 else np.nan
    return out


# ----------------------------------------------------------------------------- #
# 1. Metadata table  (one row per original image)
# ----------------------------------------------------------------------------- #
def build_metadata(data_dir, cache_path=None, recompute=False, verbose=True):
    """Scan the dataset and return a per-image DataFrame.

    Columns: filename, cls, width, height, aspect, mode, n_masks,
             intensity_{mean,std,min,max,p05,p95}, lesion_px, area_ratio,
             has_lesion, bbox_*_norm, circularity, solidity, extent,
             eccentricity, img_md5, phash, path, stem.

    If cache_path exists and recompute is False, the cached CSV is loaded
    (fast re-runs). Delete the cache or pass recompute=True to rebuild.
    """
    if cache_path and os.path.exists(cache_path) and not recompute:
        if verbose:
            print(f"[build_metadata] loading cached table: {cache_path}")
        return pd.read_csv(cache_path)

    rows = []
    for cls in CLASSES:
        d = os.path.join(data_dir, cls)
        imgs = sorted(f for f in os.listdir(d) if _NAME_RE.match(f))
        for fn in imgs:
            path = os.path.join(d, fn)
            with Image.open(path) as im:
                W, H = im.size
                mode = im.mode
                gray = np.asarray(im.convert("L"))
            rec = dict(
                filename=fn, cls=cls, path=path, stem=fn[:-4],
                width=W, height=H, aspect=W / H, mode=mode,
                n_masks=len(mask_paths_for(path)),
                intensity_mean=float(gray.mean()), intensity_std=float(gray.std()),
                intensity_min=int(gray.min()), intensity_max=int(gray.max()),
                intensity_p05=float(np.percentile(gray, 5)),
                intensity_p95=float(np.percentile(gray, 95)),
                img_md5=_md5(path), phash=str(imagehash.phash(Image.fromarray(gray))),
            )
            union = load_union_mask(path, (H, W))
            lesion_px = int(union.sum())
            rec["lesion_px"] = lesion_px
            rec["area_ratio"] = lesion_px / float(W * H)
            rec["has_lesion"] = lesion_px > 0
            if lesion_px > 0:
                ys, xs = np.where(union > 0)
                rec["bbox_x_norm"] = ((xs.min() + xs.max()) / 2) / W
                rec["bbox_y_norm"] = ((ys.min() + ys.max()) / 2) / H
                rec.update(_shape_features(union))
            else:
                for k in ["bbox_x_norm", "bbox_y_norm", "circularity",
                          "solidity", "extent", "bbox_aspect", "eccentricity"]:
                    rec[k] = np.nan
            rows.append(rec)
    df = pd.DataFrame(rows)
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        df.to_csv(cache_path, index=False, encoding="utf-8")
        if verbose:
            print(f"[build_metadata] computed {len(df)} rows -> cached to {cache_path}")
    return df


# ----------------------------------------------------------------------------- #
# 2. Overview / class distribution
# ----------------------------------------------------------------------------- #
def overview_table(df):
    """Small dataset-overview table for the notebook."""
    n = len(df)
    return pd.DataFrame({
        "property": ["# images", "# classes", "class names", "images with >=1 mask",
                     "images with a real lesion", "unique image sizes",
                     "width range (px)", "height range (px)"],
        "value": [n, df.cls.nunique(), ", ".join(CLASSES),
                  int((df.n_masks > 0).sum()), int(df.has_lesion.sum()),
                  df.groupby(["width", "height"]).ngroups,
                  f"{df.width.min()}–{df.width.max()}",
                  f"{df.height.min()}–{df.height.max()}"],
    })


def class_distribution(df):
    """Counts + percentage per class as a table."""
    t = df.cls.value_counts().reindex(CLASSES).rename("count").to_frame()
    t["percent"] = (100 * t["count"] / len(df)).round(1)
    return t.reset_index(names="class")


def plot_class_distribution(df):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    cnt = df.cls.value_counts().reindex(CLASSES)
    bars = ax[0].bar(CLASSES, cnt.values, color=[COL[c] for c in CLASSES])
    for b, v in zip(bars, cnt.values):
        ax[0].text(b.get_x() + b.get_width() / 2, v + 5,
                   f"{v}\n({v/len(df)*100:.1f}%)", ha="center", va="bottom")
    ax[0].set_ylabel("# images"); ax[0].set_title("Images per class")
    ax[0].set_ylim(0, cnt.max() * 1.2)
    ax[1].pie(cnt.values, labels=CLASSES, colors=[COL[c] for c in CLASSES],
              autopct="%1.1f%%", startangle=90, wedgeprops=dict(edgecolor="w"))
    ax[1].set_title("Class proportion")
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------- #
# 3. Sample visualization
# ----------------------------------------------------------------------------- #
def plot_sample_grid(df, n_per_class=4, seed=42):
    fig, axes = plt.subplots(len(CLASSES), n_per_class,
                             figsize=(4 * n_per_class, 4 * len(CLASSES)))
    for r, cls in enumerate(CLASSES):
        sub = df[df.cls == cls].sample(n_per_class, random_state=seed + r).reset_index(drop=True)
        for k in range(n_per_class):
            ax = axes[r, k]; row = sub.iloc[k]
            img = np.asarray(Image.open(row.path).convert("L"))
            ax.imshow(img, cmap="gray")
            union = load_union_mask(row.path, img.shape)
            if union.sum() > 0:
                cnts, _ = cv2.findContours(union, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cc in cnts:
                    ax.plot(cc[:, 0, 0], cc[:, 0, 1], color=COL[cls], lw=2)
            ax.set_title(f"{cls}  ({row.width}x{row.height})", color=COL[cls], fontsize=11)
            ax.axis("off")
    fig.suptitle("Samples with ground-truth lesion contour (normal has no lesion)", y=1.005)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------- #
# 4. Image properties: size, channels, intensity
# ----------------------------------------------------------------------------- #
def channel_summary(df):
    """PIL 'mode' distribution -> are images grayscale or RGB?"""
    return df["mode"].value_counts().rename("count").to_frame().reset_index(names="PIL_mode")


def plot_image_sizes(df):
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
    for c in CLASSES:
        s = df[df.cls == c]
        ax[0].scatter(s.width, s.height, s=12, alpha=.5, color=COL[c], label=c)
    ax[0].axvline(500, ls="--", c="k", lw=1, alpha=.4); ax[0].axhline(500, ls="--", c="k", lw=1, alpha=.4)
    ax[0].set_xlabel("width (px)"); ax[0].set_ylabel("height (px)")
    ax[0].set_title(f"Width vs height ({df.groupby(['width','height']).ngroups} unique sizes)")
    ax[0].legend()
    ax[1].hist(df.width, bins=40, color="#4c72b0", alpha=.8, label="width")
    ax[1].hist(df.height, bins=40, color="#dd8452", alpha=.7, label="height")
    ax[1].set_xlabel("pixels"); ax[1].set_title("Side-length distribution"); ax[1].legend()
    ax[2].hist(df.aspect, bins=40, color="#55a868"); ax[2].axvline(1.0, ls="--", c="k")
    ax[2].set_xlabel("aspect ratio W/H"); ax[2].set_title("Aspect ratio")
    fig.tight_layout()
    return fig


def plot_pixel_intensity(df, sample_per_class=30, seed=0):
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
    for c in CLASSES:
        sns.kdeplot(df[df.cls == c].intensity_mean, ax=ax[0], color=COL[c],
                    label=c, lw=2, fill=True, alpha=.15)
    ax[0].set_xlabel("mean gray value (0-255)"); ax[0].set_title("Per-image brightness"); ax[0].legend()
    sns.boxplot(data=df, x="cls", y="intensity_std", order=CLASSES,
                hue="cls", palette=COL, legend=False, ax=ax[1])
    ax[1].set_xlabel(""); ax[1].set_ylabel("pixel std (RMS contrast)"); ax[1].set_title("Per-image contrast")
    for c in CLASSES:
        vals = [np.asarray(Image.open(p).convert("L")).ravel()[::20]
                for p in df[df.cls == c].sample(min(sample_per_class, (df.cls == c).sum()),
                                                 random_state=seed).path]
        ax[2].hist(np.concatenate(vals), bins=64, histtype="step",
                   color=COL[c], label=c, density=True, lw=2)
    ax[2].set_xlabel("gray value"); ax[2].set_title(f"Aggregate pixel histogram\n({sample_per_class} imgs/class)")
    ax[2].legend()
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------- #
# 5. Lesion / mask analysis
# ----------------------------------------------------------------------------- #
def mask_matching_table(df):
    return pd.DataFrame({
        "check": ["images with >=1 mask", "images with 0 masks",
                  "images with multiple masks", "max masks on one image"],
        "value": [int((df.n_masks > 0).sum()), int((df.n_masks == 0).sum()),
                  int((df.n_masks > 1).sum()), int(df.n_masks.max())],
    })


def plot_mask_matching(df):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    with_m = int((df.n_masks > 0).sum()); without_m = int((df.n_masks == 0).sum())
    ax[0].bar(["with mask", "without mask"], [with_m, without_m], color=["#2ca02c", "#d62728"])
    for i, v in enumerate([with_m, without_m]):
        ax[0].text(i, v + 4, str(v), ha="center")
    ax[0].set_title("Image–mask matching")
    nm = df.n_masks.value_counts().sort_index()
    bars = ax[1].bar(nm.index.astype(str), nm.values, color="#4c72b0")
    for b, v in zip(bars, nm.values):
        ax[1].text(b.get_x() + b.get_width() / 2, v + 3, str(v), ha="center", fontsize=9)
    ax[1].set_yscale("log"); ax[1].set_xlabel("# mask files per image")
    ax[1].set_ylabel("# images (log)"); ax[1].set_title("Masks per image")
    fig.tight_layout()
    return fig


def plot_multimask_examples(df, max_examples=2):
    """Show real multi-mask images: original | each individual mask | union.

    Picks the images with the most mask files (e.g. the 3-mask cases), so the
    reader can *see* why we merge them with a pixel-wise union (load_union_mask).
    """
    multi = df[df.n_masks > 1].sort_values("n_masks", ascending=False).head(max_examples)
    if multi.empty:
        print("no multi-mask images found")
        return None
    max_m = int(df.n_masks.max())
    ncols = 1 + max_m + 1                      # original + mask slots + union
    fig, axes = plt.subplots(len(multi), ncols, figsize=(2.3 * ncols, 2.5 * len(multi)))
    axes = np.atleast_2d(axes)
    for r, row in enumerate(multi.itertuples()):
        img = cv2.cvtColor(cv2.imread(row.path), cv2.COLOR_BGR2RGB)
        axes[r, 0].imshow(img)
        axes[r, 0].set_title(f"{row.filename}\n({row.n_masks} masks)", fontsize=8)
        mps = mask_paths_for(row.path)
        for j in range(max_m):
            ax = axes[r, 1 + j]
            if j < len(mps):
                mm = cv2.imread(mps[j], cv2.IMREAD_GRAYSCALE)
                ax.imshow(mm, cmap="gray"); ax.set_title(f"mask {j + 1}", fontsize=8)
            else:
                ax.set_visible(False)
        u = load_union_mask(row.path, (row.height, row.width))
        axes[r, -1].imshow(u * 255, cmap="gray")
        axes[r, -1].set_title("union", fontsize=8, color="#2ca02c")
        for c in range(ncols):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
    fig.suptitle("Multi-mask examples — original · individual masks · pixel-wise union",
                 y=1.0, fontsize=11)
    fig.tight_layout()
    return fig


def plot_lesion_area(df):
    les = df[df.has_lesion].copy(); les["area_pct"] = les.area_ratio * 100
    bm = les[les.cls.isin(["benign", "malignant"])]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    sns.violinplot(data=bm, x="cls", y="area_pct", order=["benign", "malignant"],
                   hue="cls", palette=COL, legend=False, cut=0, inner="quartile", ax=ax[0])
    ax[0].set_xlabel(""); ax[0].set_ylabel("lesion area / image (%)"); ax[0].set_title("Lesion area ratio")
    for c in ["benign", "malignant"]:
        ax[1].hist(les[les.cls == c].area_pct, bins=40, alpha=.6, color=COL[c], label=c, density=True)
    ax[1].set_xlabel("area (%)"); ax[1].set_title("Lesion area histogram"); ax[1].legend()
    fig.tight_layout()
    return fig


def compute_location_heatmaps(df, cache_dir=None):
    """Return {class: (HMxHM freq array, n)} by summing resized binary masks."""
    hm = {c: np.zeros((HM, HM), np.float64) for c in CLASSES}
    cnt = {c: 0 for c in CLASSES}
    for row in df[df.has_lesion].itertuples():
        u = load_union_mask(row.path, (row.height, row.width))
        hm[row.cls] += cv2.resize(u, (HM, HM), interpolation=cv2.INTER_NEAREST)
        cnt[row.cls] += 1
    return {c: (hm[c] / cnt[c] if cnt[c] else hm[c], cnt[c]) for c in CLASSES}


def plot_location_heatmaps(heatmaps):
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    for i, c in enumerate(CLASSES):
        arr, n = heatmaps[c]
        im = ax[i].imshow(arr, cmap="magma", extent=[0, 1, 1, 0])
        if n == 0:
            ax[i].text(.5, .5, "no lesions\n(normal)", ha="center", va="center", color="w")
        else:
            plt.colorbar(im, ax=ax[i], fraction=.046, pad=.04, label="lesion frequency")
        ax[i].axvline(.5, ls=":", c="cyan", lw=1); ax[i].axhline(.5, ls=":", c="cyan", lw=1)
        ax[i].set_title(f"{c} (n={n})", color=COL[c])
        ax[i].set_xlabel("norm x"); ax[i].set_ylabel("norm y")
    fig.tight_layout()
    return fig


def plot_lesion_shape(df):
    les = df[df.has_lesion & df.cls.isin(["benign", "malignant"])]
    feats = [("circularity", "Circularity"), ("solidity", "Solidity"),
             ("extent", "Extent"), ("eccentricity", "Eccentricity")]
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.5))
    for i, (f, t) in enumerate(feats):
        sns.violinplot(data=les, x="cls", y=f, order=["benign", "malignant"],
                       hue="cls", palette=COL, legend=False, cut=0, inner="quartile", ax=ax[i])
        ax[i].set_title(t); ax[i].set_xlabel(""); ax[i].set_ylabel("")
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------- #
# 6. Data quality: exact / near duplicates and split leakage
# ----------------------------------------------------------------------------- #
def exact_duplicates(df):
    """Groups of byte-identical images (same MD5)."""
    g = df.groupby("img_md5")
    out = []
    for h, idx in g.groups.items():
        if len(idx) > 1:
            sub = df.loc[idx]
            out.append(dict(md5=h, n=len(idx),
                            files=", ".join(sub.filename),
                            classes=", ".join(sorted(sub.cls.unique()))))
    return pd.DataFrame(out)


def find_near_duplicates(df, thresh=5):
    """Return (near_pairs_df, clusters). Uses pHash Hamming distance (vectorized)
    and union-find to group transitively-similar images into clusters."""
    sub = df[df.phash.notna()].reset_index(drop=True)
    fns = sub.filename.tolist()
    cls = sub.cls.tolist()
    # 64-bit pHash -> (N,64) bit matrix
    bits = np.array([[int(b) for b in bin(int(str(h), 16))[2:].zfill(64)]
                     for h in sub.phash], dtype=np.uint8)
    N = len(fns)
    pairs = []
    for i in range(N):                       # row-wise to keep memory small
        d = (bits[i] != bits[i + 1:]).sum(1)
        for off in np.where(d <= thresh)[0]:
            j = i + 1 + off
            pairs.append((fns[i], fns[j], int(d[off]), cls[i], cls[j]))
    near = pd.DataFrame(pairs, columns=["file_a", "file_b", "hamming", "cls_a", "cls_b"])
    # union-find clusters
    parent = {f: f for f in fns}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for a, b, *_ in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    clusters = collections.defaultdict(list)
    for f in fns:
        clusters[find(f)].append(f)
    clusters = {k: v for k, v in clusters.items() if len(v) > 1}
    return near, clusters


def plot_duplicates(df, near, thresh=5):
    fig = plt.figure(figsize=(15, 5.2))
    # left: per-image nearest-neighbour distance
    sub = df[df.phash.notna()].reset_index(drop=True)
    bits = np.array([[int(b) for b in bin(int(str(h), 16))[2:].zfill(64)]
                     for h in sub.phash], dtype=np.uint8)
    mind = []
    for i in range(len(bits)):
        d = (bits[i] != bits).sum(1); d[i] = 999; mind.append(int(d.min()))
    axL = fig.add_axes([0.06, 0.12, 0.4, 0.75])
    axL.hist(mind, bins=range(0, 33), color="#4c72b0", edgecolor="w")
    for t in (1, thresh, 10):
        axL.axvline(t, ls="--", c="r", alpha=.5)
    axL.set_xlabel("min Hamming distance to any other image")
    axL.set_ylabel("# images"); axL.set_title("Nearest-neighbour pHash distance")
    # right: example near-duplicate pairs (put a cross-class one first)
    fig.text(0.77, 0.93, "Example near-duplicate pairs", ha="center", fontsize=12, weight="bold")
    ns = near.sort_values("hamming"); show = []
    cross = ns[ns.cls_a != ns.cls_b]
    if len(cross):
        show.append(cross.iloc[0])
    for _, r in ns.iterrows():
        if len(show) >= 3:
            break
        if not any((r.file_a == s.file_a and r.file_b == s.file_b) for s in show):
            show.append(r)
    gs = fig.add_gridspec(len(show), 2, left=0.56, right=0.98, top=0.88, bottom=0.06,
                          hspace=0.5, wspace=0.05)
    look = df.set_index("filename")
    for i, r in enumerate(show):
        for jj, fn in enumerate([r.file_a, r.file_b]):
            rr = look.loc[fn]; a = fig.add_subplot(gs[i, jj])
            a.imshow(np.asarray(Image.open(rr.path).convert("L")), cmap="gray"); a.axis("off")
            tag = rr.cls + ("  [CROSS-CLASS]" if r.cls_a != r.cls_b else "")
            a.set_title(f"{tag}  Ham={r.hamming}", fontsize=9, color=COL[rr.cls])
    return fig


def simulate_split_leakage(df, clusters, n_sims=40, test_frac=0.30, seed=0):
    """How many near-duplicate clusters get split across train/test under a
    naive RANDOM split (averaged over n_sims). 0 would mean 'no leakage'."""
    rng = np.random.RandomState(seed)
    files = df.filename.tolist(); N = len(files)
    member_clusters = [v for v in clusters.values() if len(v) > 1]
    leaks = []
    for _ in range(n_sims):
        idx = rng.permutation(N)
        train = set(np.array(files)[idx[int(test_frac * N):]])
        leaked = 0
        for members in member_clusters:
            ins = [m in train for m in members]
            if any(ins) and not all(ins):
                leaked += 1
        leaks.append(leaked)
    return leaks


def plot_leakage(df, clusters, leaks):
    sizes = sorted((len(v) for v in clusters.values() if len(v) > 1), reverse=True)
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
    ax[0].bar(range(len(sizes)), sizes, color="#c44e52")
    ax[0].set_xlabel("near-duplicate cluster (sorted)"); ax[0].set_ylabel("# images in cluster")
    ax[0].set_title(f"{len(sizes)} clusters, {sum(sizes)} images "
                    f"({sum(sizes)/len(df)*100:.0f}% of data)")
    ax[1].hist(leaks, bins=range(min(leaks), max(leaks) + 2), color="#dd8452", edgecolor="w")
    ax[1].axvline(0, ls="--", c="g", lw=2)
    ax[1].set_xlabel("# clusters split across train/test")
    ax[1].set_ylabel(f"frequency ({len(leaks)} random splits)")
    ax[1].set_title(f"Leakage under RANDOM split (mean {np.mean(leaks):.0f})")
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------- #
# 7. Preprocessing demonstration
# ----------------------------------------------------------------------------- #
def _letterbox(img, size=256):
    """Resize keeping aspect ratio, pad to a square (no lesion distortion)."""
    h, w = img.shape[:2]
    s = size / max(h, w)
    nh, nw = int(round(h * s)), int(round(w * s))
    r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size), np.uint8)
    y0, x0 = (size - nh) // 2, (size - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = r
    return canvas


def plot_preprocessing_demo(df, size=256, seed=7):
    """Show the candidate preprocessing steps on one representative lesion image."""
    row = df[df.cls == "malignant"].sample(1, random_state=seed).iloc[0]
    g = np.asarray(Image.open(row.path).convert("L"))
    stretch = cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA)      # naive resize
    letter = _letterbox(g, size)                                            # resize+pad
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(letter)  # contrast
    denoise = cv2.fastNlMeansDenoising(letter, h=10)                        # speckle
    panels = [(g, f"original {g.shape[1]}x{g.shape[0]}"),
              (stretch, f"resize {size}x{size}\n(distorts aspect)"),
              (letter, f"resize+pad {size}x{size}\n(keeps aspect)"),
              (clahe, "CLAHE on padded\n(contrast boost)"),
              (denoise, "NLM denoise\n(speckle removal)")]
    fig, ax = plt.subplots(1, 5, figsize=(20, 4.2))
    for a, (im, t) in zip(ax, panels):
        a.imshow(im, cmap="gray"); a.set_title(t, fontsize=10); a.axis("off")
    fig.suptitle(f"Candidate preprocessing on one {row.cls} image", y=1.02)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------- #
# Summary numbers (for the closing markdown / no invented values)
# ----------------------------------------------------------------------------- #
def summary_stats(df, near=None, clusters=None):
    s = dict(
        n=len(df),
        per_class={c: int((df.cls == c).sum()) for c in CLASSES},
        pct={c: round(100 * (df.cls == c).mean(), 1) for c in CLASSES},
        unique_sizes=df.groupby(["width", "height"]).ngroups,
        w_range=(int(df.width.min()), int(df.width.max())),
        h_range=(int(df.height.min()), int(df.height.max())),
        multi_mask=int((df.n_masks > 1).sum()),
        area_med={c: round(float(df[(df.cls == c) & df.has_lesion].area_ratio.median()) * 100, 1)
                  for c in ["benign", "malignant"]},
        modes=df["mode"].value_counts().to_dict(),
    )
    if near is not None:
        s["near_pairs"] = int(len(near))
        s["near_cross"] = int((near.cls_a != near.cls_b).sum()) if len(near) else 0
    if clusters is not None:
        s["clusters"] = len(clusters)
        s["imgs_in_clusters"] = int(sum(len(v) for v in clusters.values()))
    return s
