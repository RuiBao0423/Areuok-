# -*- coding: utf-8 -*-
"""BUSI EDA - Step 2: produce all figures into EDA/figures/ from metadata + artifacts."""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
import cv2
from PIL import Image

sns.set_theme(style="whitegrid", context="talk")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART  = os.path.join(ROOT, "artifacts")
FIG  = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

df = pd.read_csv(os.path.join(ART, "metadata.csv"))
summary = json.load(open(os.path.join(ART, "summary.json"), encoding="utf-8"))
CLS = ["benign", "malignant", "normal"]
COL = {"benign": "#2ca02c", "malignant": "#d62728", "normal": "#7f7f7f"}
def save(fig, name):
    fig.tight_layout(); fig.savefig(os.path.join(FIG, name), dpi=130, bbox_inches="tight"); plt.close(fig)
    print("saved", name)

# ---------- 1. Class distribution ----------
fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
cnt = df.cls.value_counts().reindex(CLS)
bars = ax[0].bar(CLS, cnt.values, color=[COL[c] for c in CLS])
for b, v in zip(bars, cnt.values):
    ax[0].text(b.get_x()+b.get_width()/2, v+6, f"{v}\n({v/len(df)*100:.1f}%)", ha="center", va="bottom", fontsize=13)
ax[0].set_title("Class distribution (n=780)"); ax[0].set_ylabel("# images"); ax[0].set_ylim(0, cnt.max()*1.18)
ax[1].pie(cnt.values, labels=CLS, colors=[COL[c] for c in CLS], autopct="%1.1f%%",
          startangle=90, wedgeprops=dict(edgecolor="w"))
ax[1].set_title("Class proportion")
fig.suptitle("1. Class Distribution  —  imbalanced (benign : malignant : normal ≈ 3.3 : 1.6 : 1)", y=1.03, fontsize=15)
save(fig, "fig01_class_distribution.png")

# ---------- 2. Image size & aspect ratio ----------
fig, ax = plt.subplots(1, 3, figsize=(19, 5.5))
for c in CLS:
    s = df[df.cls == c]
    ax[0].scatter(s.width, s.height, s=14, alpha=.5, color=COL[c], label=c)
ax[0].axvline(500, ls="--", c="k", lw=1, alpha=.5); ax[0].axhline(500, ls="--", c="k", lw=1, alpha=.5)
ax[0].set_xlabel("width (px)"); ax[0].set_ylabel("height (px)")
ax[0].set_title(f"Width vs Height\n{summary['unique_sizes']} unique sizes / 780 images"); ax[0].legend()
ax[1].hist(df.width, bins=40, color="#4c72b0", alpha=.8, label="width")
ax[1].hist(df.height, bins=40, color="#dd8452", alpha=.7, label="height")
ax[1].set_title("Side-length distribution"); ax[1].set_xlabel("pixels"); ax[1].legend()
ax[2].hist(df.aspect, bins=40, color="#55a868")
ax[2].axvline(1.0, ls="--", c="k"); ax[2].set_title("Aspect ratio (W/H)"); ax[2].set_xlabel("W / H")
fig.suptitle("2. Image Size & Aspect-Ratio Distribution  —  NOT uniform; resizing required before training", y=1.03, fontsize=15)
save(fig, "fig02_image_size.png")

# ---------- 3. Sample visualization (image + mask overlay) ----------
DATA_dir = df.iloc[0].path  # just to reuse path building
fig, axes = plt.subplots(3, 4, figsize=(18, 12))
rng = np.random.RandomState(42)
for r, c in enumerate(CLS):
    sub = df[df.cls == c].sample(min(4, (df.cls==c).sum()), random_state=r+1).reset_index(drop=True)
    for k in range(4):
        ax = axes[r, k]
        row = sub.iloc[k]
        img = np.asarray(Image.open(row.path).convert("L"))
        ax.imshow(img, cmap="gray")
        # overlay mask contour
        d = os.path.dirname(row.path)
        union = np.zeros(img.shape, np.uint8)
        for f in os.listdir(d):
            if f.startswith(row.stem + "_mask") and f.endswith(".png"):
                mm = cv2.imread(os.path.join(d, f), cv2.IMREAD_GRAYSCALE)
                if mm is not None:
                    if mm.shape != img.shape: mm = cv2.resize(mm, img.shape[::-1], interpolation=cv2.INTER_NEAREST)
                    union |= (mm > 127).astype(np.uint8)
        if union.sum() > 0:
            cnts,_ = cv2.findContours(union, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cc in cnts:
                ax.plot(cc[:,0,0], cc[:,0,1], color=COL[c], lw=2)
        ax.set_title(f"{c}  ({row.width}x{row.height})", fontsize=12, color=COL[c])
        ax.axis("off")
fig.suptitle("3. Sample Visualization  —  ultrasound image with ground-truth lesion contour (normal = no lesion)", y=1.01, fontsize=16)
save(fig, "fig03_samples.png")

# ---------- 4. Mask availability & image-mask matching ----------
fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
avail = pd.DataFrame({"with mask":[summary["images_with_mask"]], "without mask":[summary["images_without_mask"]]})
ax[0].bar(["with mask","without mask"], [summary["images_with_mask"], summary["images_without_mask"]],
          color=["#2ca02c","#d62728"])
ax[0].set_title(f"Image–mask matching\n{summary['images_with_mask']}/780 images have ≥1 mask (100%)")
for i,v in enumerate([summary["images_with_mask"], summary["images_without_mask"]]):
    ax[0].text(i, v+5, str(v), ha="center", fontsize=13)
nm = df.n_masks.value_counts().sort_index()
bars = ax[1].bar(nm.index.astype(str), nm.values, color="#4c72b0")
for b,v in zip(bars, nm.values): ax[1].text(b.get_x()+b.get_width()/2, v+5, str(v), ha="center", fontsize=12)
ax[1].set_title(f"# masks per image  (multi-mask: {summary['multi_mask_images']} imgs, max {summary['max_masks_per_image']})")
ax[1].set_xlabel("number of mask files"); ax[1].set_ylabel("# images"); ax[1].set_yscale("log")
fig.suptitle("4. Mask Availability & Image–Mask Matching  —  17 images have multiple lesion masks (16 benign, 1 malignant)", y=1.03, fontsize=14)
save(fig, "fig04_mask_matching.png")

# ---------- 5. Lesion area ratio ----------
les = df[df.has_lesion].copy()
les["area_pct"] = les.area_ratio*100
fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
sns.violinplot(data=les[les.cls.isin(["benign","malignant"])], x="cls", y="area_pct",
               palette={k:COL[k] for k in ["benign","malignant"]}, ax=ax[0], cut=0, inner="quartile")
ax[0].set_title("Lesion area ratio by class"); ax[0].set_ylabel("lesion area / image area (%)"); ax[0].set_xlabel("")
for c in ["benign","malignant"]:
    ax[1].hist(les[les.cls==c].area_pct, bins=40, alpha=.6, color=COL[c], label=c, density=True)
ax[1].set_title("Lesion area ratio histogram"); ax[1].set_xlabel("area (%)"); ax[1].legend()
mb = summary["area_ratio_median_by_class"]
fig.suptitle(f"5. Lesion Area Ratio  —  malignant lesions much larger (median {mb['malignant']*100:.1f}% vs benign {mb['benign']*100:.1f}%)", y=1.03, fontsize=14)
save(fig, "fig05_area_ratio.png")

# ---------- 6. Lesion location heatmap ----------
fig, ax = plt.subplots(1, 3, figsize=(18, 6))
for i, c in enumerate(["benign","malignant","normal"]):
    hm = np.load(os.path.join(ART, f"heatmap_{c}.npy"))
    n = summary_counts = int(((df.cls==c) & df.has_lesion).sum())
    if n > 0:
        hm = hm / n
        im = ax[i].imshow(hm, cmap="magma", extent=[0,1,1,0])
        plt.colorbar(im, ax=ax[i], fraction=.046, pad=.04, label="lesion frequency")
    else:
        ax[i].imshow(np.zeros((256,256)), cmap="magma", extent=[0,1,1,0])
        ax[i].text(.5,.5,"no lesions\n(normal)", ha="center", va="center", color="w", fontsize=14)
    ax[i].axvline(.5, ls=":", c="cyan", lw=1); ax[i].axhline(.5, ls=":", c="cyan", lw=1)
    ax[i].set_title(f"{c}  (n={n})", color=COL[c]); ax[i].set_xlabel("normalized x"); ax[i].set_ylabel("normalized y")
fig.suptitle("6. Lesion Location Heatmap  —  lesions concentrate near image centre (acquisition bias)", y=1.02, fontsize=15)
save(fig, "fig06_location_heatmap.png")

# ---------- 7. Pixel intensity / contrast ----------
fig, ax = plt.subplots(1, 3, figsize=(19, 5.5))
for c in CLS:
    sns.kdeplot(df[df.cls==c].intensity_mean, ax=ax[0], color=COL[c], label=c, lw=2, fill=True, alpha=.15)
ax[0].set_title("Mean brightness per image"); ax[0].set_xlabel("mean gray value (0-255)"); ax[0].legend()
sns.boxplot(data=df, x="cls", y="intensity_std", order=CLS, palette=COL, ax=ax[1])
ax[1].set_title("Contrast (pixel std) by class"); ax[1].set_ylabel("RMS contrast (std)"); ax[1].set_xlabel("")
# representative aggregate intensity histogram (sample a few images per class)
for c in CLS:
    vals = []
    for p in df[df.cls==c].sample(min(30,(df.cls==c).sum()), random_state=0).path:
        vals.append(np.asarray(Image.open(p).convert("L")).ravel()[::20])
    vals = np.concatenate(vals)
    ax[2].hist(vals, bins=64, histtype="step", color=COL[c], label=c, density=True, lw=2)
ax[2].set_title("Aggregate pixel-intensity histogram\n(30 imgs/class sample)"); ax[2].set_xlabel("gray value"); ax[2].legend()
fig.suptitle("7. Pixel Intensity / Contrast Distribution  —  classes overlap heavily → normalization needed, intensity alone is weak", y=1.03, fontsize=13)
save(fig, "fig07_intensity.png")

# ---------- 8. Lesion shape analysis ----------
feats = [("circularity","Circularity 4πA/P²"),("solidity","Solidity A/hull"),
         ("extent","Extent A/bbox"),("eccentricity","Eccentricity")]
fig, ax = plt.subplots(1, 4, figsize=(22, 5.5))
bm = les[les.cls.isin(["benign","malignant"])]
for i,(f,t) in enumerate(feats):
    sns.violinplot(data=bm, x="cls", y=f, palette={k:COL[k] for k in ["benign","malignant"]},
                   ax=ax[i], cut=0, inner="quartile")
    ax[i].set_title(t); ax[i].set_xlabel(""); ax[i].set_ylabel("")
fig.suptitle("8. Lesion Shape Analysis  —  malignant lesions are less circular / less regular (more irregular boundaries)", y=1.03, fontsize=15)
save(fig, "fig08_shape.png")

# ---------- 9. Duplicate / near-duplicate detection ----------
import imagehash
near = pd.read_csv(os.path.join(ART, "near_duplicates.csv"))
# per-image nearest neighbor distance
hashes = {r.filename: imagehash.hex_to_hash(r.phash) for r in df.itertuples() if isinstance(r.phash,str)}
fns = list(hashes.keys()); arr = np.array([[int(b) for b in bin(int(str(hashes[f]),16))[2:].zfill(64)] for f in fns])
# compute min hamming to any other
mind = []
for i in range(len(fns)):
    d = (arr[i] != arr).sum(1); d[i] = 999; mind.append(d.min())
fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
ax[0].hist(mind, bins=range(0,33), color="#4c72b0", edgecolor="w")
for t in [1,5,10]:
    ax[0].axvline(t, ls="--", c="r", alpha=.5); ax[0].text(t,ax[0].get_ylim()[1]*.9,f"≤{t}",color="r")
ax[0].set_title(f"Nearest-neighbour pHash distance\n{summary['near_dup_pairs']} pairs ≤5, {summary['near_dup_images_involved']} imgs in {summary['near_dup_clusters']} clusters")
ax[0].set_xlabel("min Hamming distance"); ax[0].set_ylabel("# images")
# montage of top near-dup pairs (incl a cross-class one if present)
near_sorted = near.sort_values("hamming")
show = []
cross = near_sorted[near_sorted.cls_a != near_sorted.cls_b]
if len(cross): show.append(cross.iloc[0])
for _,r in near_sorted.iterrows():
    if len(show) >= 3: break
    if not any(r.file_a==s.file_a and r.file_b==s.file_b for s in show): show.append(r)
ax[1].axis("off")
fig.text(.77, .90, "Example near-duplicate pairs (pHash)", ha="center", fontsize=13, weight="bold")
inner = fig.add_gridspec(len(show), 2, left=.56, right=.98, top=.82, bottom=.06, hspace=.45, wspace=.05)
def find_path(fn):
    row = df[df.filename==fn].iloc[0]; return row.path, row.cls
for i,r in enumerate(show):
    for j,fn in enumerate([r.file_a, r.file_b]):
        p,cl = find_path(fn); a = fig.add_subplot(inner[i,j])
        a.imshow(np.asarray(Image.open(p).convert("L")), cmap="gray"); a.axis("off")
        tag = f"{cl}" + ("  [CROSS-CLASS]" if r.cls_a!=r.cls_b else "")
        a.set_title(f"{tag}   Ham={r.hamming}", fontsize=9, color=COL[cl])
fig.suptitle("9. Duplicate / Near-Duplicate Detection  —  many repeated/near-identical frames; 10 pairs even cross classes", y=1.02, fontsize=14)
save(fig, "fig09_duplicates.png")

# ---------- 10. Train-test leakage check ----------
# cluster sizes
import collections
# rebuild clusters from near pairs (union-find)
parent={f:f for f in fns}
def find(x):
    while parent[x]!=x: parent[x]=parent[parent[x]]; x=parent[x]
    return x
for _,r in near.iterrows():
    a,b=find(r.file_a),find(r.file_b)
    if a!=b: parent[a]=b
clu=collections.defaultdict(list)
for f in fns: clu[find(f)].append(f)
sizes=sorted([len(v) for v in clu.values() if len(v)>1], reverse=True)

fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
ax[0].bar(range(len(sizes)), sizes, color="#c44e52")
ax[0].set_title(f"Near-duplicate cluster sizes\n{len(sizes)} clusters, {sum(sizes)} images affected ({sum(sizes)/780*100:.0f}% of data)")
ax[0].set_xlabel("cluster (sorted)"); ax[0].set_ylabel("# images in cluster")
# simulate random split leakage vs grouped
rng=np.random.RandomState(0); N=40; leaks=[]
allfiles=list(df.filename)
file2clu={f:find(f) for f in fns}
for _ in range(N):
    idx=rng.permutation(len(allfiles)); tr=set(np.array(allfiles)[idx[:int(.7*len(allfiles))]])
    te=set(np.array(allfiles)[idx[int(.7*len(allfiles)):]])
    leaked=0
    for cid,members in clu.items():
        if len(members)<2: continue
        inss=[m in tr for m in members]
        if any(inss) and not all(inss): leaked+=1
    leaks.append(leaked)
ax[1].hist(leaks, bins=range(min(leaks),max(leaks)+2), color="#dd8452", edgecolor="w")
ax[1].axvline(0, ls="--", c="g", lw=2)
ax[1].set_title(f"Leaked clusters under RANDOM 70/30 split\n(mean {np.mean(leaks):.0f} clusters split across train/test)")
ax[1].set_xlabel("# near-dup clusters split across train & test"); ax[1].set_ylabel("frequency (40 sims)")
fig.suptitle("10. Train–Test Leakage Check  —  random split leaks ~"+f"{np.mean(leaks):.0f}"+" clusters → use GROUPED (cluster/patient-aware) splitting", y=1.03, fontsize=13)
save(fig, "fig10_leakage.png")

print("\nALL FIGURES DONE ->", FIG)
print("mean leaked clusters (random split):", round(float(np.mean(leaks)),1))
