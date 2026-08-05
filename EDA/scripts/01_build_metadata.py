# -*- coding: utf-8 -*-
"""
BUSI EDA - Step 1: scan dataset, build per-image metadata table + artifacts.
Outputs (EDA/artifacts/):
  metadata.csv        one row per ORIGINAL image (masks joined)
  heatmap_<cls>.npy   summed resized binary masks per class (256x256)
  summary.json        aggregate statistics used by report/ppt
"""
import os, re, json, hashlib
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import imagehash

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # .../EDA
PROJ = os.path.dirname(ROOT)
DATA = os.path.join(PROJ, "datasets", "Breast-Cancer-Ultrasound-Images-Dataset", "Dataset_BUSI_with_GT")
ART  = os.path.join(ROOT, "artifacts")
os.makedirs(ART, exist_ok=True)

CLASSES = ["benign", "malignant", "normal"]
HM = 256  # heatmap / shape resample resolution
name_re = re.compile(r"^(benign|malignant|normal) \((\d+)\)\.png$", re.IGNORECASE)

def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()

records = []
heatmaps = {c: np.zeros((HM, HM), np.float64) for c in CLASSES}
heat_counts = {c: 0 for c in CLASSES}

for cls in CLASSES:
    d = os.path.join(DATA, cls)
    files = os.listdir(d)
    imgs = sorted([f for f in files if name_re.match(f)])
    for fn in imgs:
        m = name_re.match(fn)
        idx = int(m.group(2))
        ipath = os.path.join(d, fn)
        stem = fn[:-4]  # drop .png
        # collect masks: "<stem>_mask.png", "<stem>_mask_1.png", ...
        masks = [f for f in files if f.startswith(stem + "_mask") and f.endswith(".png")]
        masks = sorted(masks)

        # ---- image ----
        with Image.open(ipath) as im:
            W, H = im.size
            mode = im.mode
            gray = np.asarray(im.convert("L"))
        fsize_kb = os.path.getsize(ipath) / 1024.0
        rec = dict(
            filename=fn, cls=cls, idx=idx, path=ipath, stem=stem,
            width=W, height=H, aspect=W / H, mode=mode, filesize_kb=fsize_kb,
            n_masks=len(masks),
            intensity_mean=float(gray.mean()),
            intensity_std=float(gray.std()),           # RMS contrast
            intensity_min=int(gray.min()),
            intensity_max=int(gray.max()),
            intensity_p05=float(np.percentile(gray, 5)),
            intensity_p95=float(np.percentile(gray, 95)),
            img_md5=md5(ipath),
        )
        try:
            rec["phash"] = str(imagehash.phash(Image.open(ipath).convert("L")))
        except Exception:
            rec["phash"] = None

        # ---- union of masks ----
        union = np.zeros((H, W), np.uint8)
        for mk in masks:
            mm = cv2.imread(os.path.join(d, mk), cv2.IMREAD_GRAYSCALE)
            if mm is None:
                continue
            if mm.shape != (H, W):
                mm = cv2.resize(mm, (W, H), interpolation=cv2.INTER_NEAREST)
            union |= (mm > 127).astype(np.uint8)

        lesion_px = int(union.sum())
        rec["lesion_px"] = lesion_px
        rec["area_ratio"] = lesion_px / float(W * H)
        rec["has_lesion"] = lesion_px > 0

        if lesion_px > 0:
            ys, xs = np.where(union > 0)
            x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
            rec["bbox_x_norm"] = ((x0 + x1) / 2) / W
            rec["bbox_y_norm"] = ((y0 + y1) / 2) / H
            rec["bbox_w_norm"] = (x1 - x0 + 1) / W
            rec["bbox_h_norm"] = (y1 - y0 + 1) / H
            # largest contour shape features
            cnts, _ = cv2.findContours(union, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            c = max(cnts, key=cv2.contourArea)
            area = cv2.contourArea(c)
            peri = cv2.arcLength(c, True)
            hull = cv2.convexHull(c)
            harea = cv2.contourArea(hull)
            bx, by, bw, bh = cv2.boundingRect(c)
            rec["circularity"] = float(4 * np.pi * area / (peri ** 2)) if peri > 0 else np.nan
            rec["solidity"]    = float(area / harea) if harea > 0 else np.nan
            rec["extent"]      = float(area / (bw * bh)) if bw * bh > 0 else np.nan
            rec["bbox_aspect"] = float(bw / bh) if bh > 0 else np.nan
            rec["equiv_diam_norm"] = float(np.sqrt(4 * area / np.pi) / np.sqrt(W * H))
            if len(c) >= 5:
                (_, _), (MA, ma), _ = cv2.fitEllipse(c)
                a_, b_ = max(MA, ma) / 2.0, min(MA, ma) / 2.0
                rec["eccentricity"] = float(np.sqrt(1 - (b_ ** 2) / (a_ ** 2))) if a_ > 0 else np.nan
            else:
                rec["eccentricity"] = np.nan
            # accumulate heatmap
            rmask = cv2.resize(union, (HM, HM), interpolation=cv2.INTER_NEAREST)
            heatmaps[cls] += rmask
            heat_counts[cls] += 1
        else:
            for k in ["bbox_x_norm","bbox_y_norm","bbox_w_norm","bbox_h_norm",
                      "circularity","solidity","extent","bbox_aspect",
                      "equiv_diam_norm","eccentricity"]:
                rec[k] = np.nan

        records.append(rec)

df = pd.DataFrame.from_records(records)
df.to_csv(os.path.join(ART, "metadata.csv"), index=False, encoding="utf-8")
for c in CLASSES:
    np.save(os.path.join(ART, f"heatmap_{c}.npy"), heatmaps[c])

# ---------- exact + near duplicate detection ----------
# exact
dup_exact = (df.groupby("img_md5").size().reset_index(name="n"))
exact_groups = dup_exact[dup_exact.n > 1]
# near-dup via pHash Hamming distance
hashes = {r.filename: imagehash.hex_to_hash(r.phash) for r in df.itertuples() if r.phash}
fns = list(hashes.keys())
THRESH = 5
near_pairs = []
for i in range(len(fns)):
    hi = hashes[fns[i]]
    for j in range(i + 1, len(fns)):
        dist = hi - hashes[fns[j]]
        if dist <= THRESH:
            ci = df.loc[df.filename == fns[i], "cls"].values[0]
            cj = df.loc[df.filename == fns[j], "cls"].values[0]
            near_pairs.append((fns[i], fns[j], int(dist), ci, cj))
near_df = pd.DataFrame(near_pairs, columns=["file_a","file_b","hamming","cls_a","cls_b"])
near_df.to_csv(os.path.join(ART, "near_duplicates.csv"), index=False, encoding="utf-8")

# union-find clusters on near-dup pairs (for leakage / grouped split)
parent = {f: f for f in fns}
def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]; x = parent[x]
    return x
def union_(a, b):
    ra, rb = find(a), find(b)
    if ra != rb: parent[ra] = rb
for a, b, *_ in near_pairs:
    union_(a, b)
clusters = {}
for f in fns:
    clusters.setdefault(find(f), []).append(f)
multi_clusters = {k: v for k, v in clusters.items() if len(v) > 1}

summary = dict(
    total_images=int(len(df)),
    per_class={c: int((df.cls == c).sum()) for c in CLASSES},
    per_class_pct={c: round(100 * (df.cls == c).mean(), 2) for c in CLASSES},
    images_with_mask=int((df.n_masks > 0).sum()),
    images_without_mask=int((df.n_masks == 0).sum()),
    multi_mask_images=int((df.n_masks > 1).sum()),
    multi_mask_by_class={c: int(((df.cls == c) & (df.n_masks > 1)).sum()) for c in CLASSES},
    max_masks_per_image=int(df.n_masks.max()),
    images_with_lesion=int(df.has_lesion.sum()),
    normal_with_lesion=int(((df.cls=="normal") & (df.has_lesion)).sum()),
    width_range=[int(df.width.min()), int(df.width.max())],
    height_range=[int(df.height.min()), int(df.height.max())],
    width_median=float(df.width.median()),
    height_median=float(df.height.median()),
    unique_sizes=int(df.groupby(["width","height"]).ngroups),
    area_ratio_median_by_class={c: round(float(df[(df.cls==c)&df.has_lesion].area_ratio.median()),4)
                                for c in ["benign","malignant"]},
    area_ratio_mean_by_class={c: round(float(df[(df.cls==c)&df.has_lesion].area_ratio.mean()),4)
                              for c in ["benign","malignant"]},
    intensity_mean_by_class={c: round(float(df[df.cls==c].intensity_mean.mean()),2) for c in CLASSES},
    intensity_std_by_class={c: round(float(df[df.cls==c].intensity_std.mean()),2) for c in CLASSES},
    exact_dup_groups=int(len(exact_groups)),
    exact_dup_images=int(exact_groups.n.sum()) if len(exact_groups) else 0,
    near_dup_pairs=int(len(near_df)),
    near_dup_pairs_cross_class=int((near_df.cls_a != near_df.cls_b).sum()) if len(near_df) else 0,
    near_dup_clusters=int(len(multi_clusters)),
    near_dup_images_involved=int(sum(len(v) for v in multi_clusters.values())),
    phash_threshold=THRESH,
)
with open(os.path.join(ART, "summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print(json.dumps(summary, indent=2, ensure_ascii=False))
print("\nSaved metadata.csv rows:", len(df))
