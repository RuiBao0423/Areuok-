# EDA 详细说明 —— 每一张图、每一段代码

> 目标读者：想搞懂"这份 EDA 到底怎么做出来的"。本文逐段讲解代码逻辑，再逐张讲解 10 张图（每张图 = 画了什么 / 代码怎么算 / 怎么读 / 结论）。
> 数据集：`datasets/Breast-Cancer-Ultrasound-Images-Dataset/Dataset_BUSI_with_GT/`（benign / malignant / normal 三个文件夹）。

---

## 0. 整体流程

EDA 分成 3 个脚本，依次运行，前一个的产物喂给后一个：

```
01_build_metadata.py   扫描 780 张图+mask → 算出每张图的所有特征
        │  产物：artifacts/metadata.csv、summary.json、near_duplicates.csv、heatmap_*.npy
        ▼
02_make_figures.py     读上面的产物 → 画 10 张图
        │  产物：figures/fig01 ~ fig10.png
        ▼
03_build_ppt.py        把 10 张图 + 结论 → 拼成幻灯片
           产物：BUSI_EDA.pptx
```

**核心设计思想**：先把"读图/算特征"这种慢操作做一次，把结果落盘成一张表（`metadata.csv`，780 行 × 30 多列，一行 = 一张原图）。之后所有画图都只读这张表，不再反复读原图，既快又保证每张图用的是同一份数据。

### 数据集的文件结构（先搞懂这个，代码才好懂）

每个类别文件夹里，一张病例有 1 张原图 + 1~3 张 mask：

```
benign (1).png          ← 原图（超声灰度图）
benign (1)_mask.png     ← 该病灶的二值 mask（白=病灶，黑=背景）
benign (1)_mask_1.png   ← 少数图有第二、第三个病灶 → 多出来的 mask
...
```

- **原图**：文件名形如 `类别 (数字).png`，不含 `_mask`。
- **mask**：在原图名后面加 `_mask`（或 `_mask_1`、`_mask_2`）。
- **normal 类**：也有 mask 文件，但整张全黑（没有病灶）。

---

## 1. `01_build_metadata.py` 逐段讲解

### 1.1 路径、常量、文件名正则

```python
CLASSES = ["benign", "malignant", "normal"]
HM = 256   # 热力图/形状统一缩放到 256×256
name_re = re.compile(r"^(benign|malignant|normal) \((\d+)\)\.png$", re.IGNORECASE)
```

- `name_re` 是关键：它**只匹配原图**（`benign (1).png`），不会匹配 mask（因为 mask 名里有 `_mask`，正则末尾 `\.png$` 前不允许有 `_mask`）。这样遍历时就能把"原图"和"它的 mask"分开处理。
- `(\d+)` 捕获括号里的编号，用来记录 `idx`。

### 1.2 MD5：找"字节完全相同"的重复图

```python
def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()
```

- 把文件按二进制分块读入，算 MD5 指纹。
- 两张图 MD5 相同 ⟺ 文件**逐字节相同**（100% 是同一张图）。这是"精确重复"检测，后面用它抓出了那对 benign/malignant 冲突图。

### 1.3 主循环：把每张原图配上它的 mask

```python
for cls in CLASSES:
    files = os.listdir(d)                       # 该类别下所有文件
    imgs = sorted([f for f in files if name_re.match(f)])   # 只留原图
    for fn in imgs:
        stem = fn[:-4]                          # 去掉 ".png"，得到 "benign (1)"
        masks = [f for f in files if f.startswith(stem + "_mask") and f.endswith(".png")]
```

- `masks` 用"以 `stem+_mask` 开头"来收集该图的所有 mask，天然支持一图多 mask（`_mask`、`_mask_1`、`_mask_2`）。

### 1.4 图像层面的特征（尺寸 + 强度）

```python
with Image.open(ipath) as im:
    W, H = im.size
    gray = np.asarray(im.convert("L"))          # 转灰度
rec = dict(width=W, height=H, aspect=W/H, n_masks=len(masks),
           intensity_mean=float(gray.mean()),
           intensity_std=float(gray.std()),      # RMS 对比度
           intensity_min=..., intensity_max=..., 
           intensity_p05=np.percentile(gray,5), intensity_p95=np.percentile(gray,95))
```

- `aspect = W/H`：宽高比，>1 = 横图。
- `intensity_mean`：整张图平均亮度（0–255）。
- `intensity_std`：像素值标准差，即 **RMS 对比度**（越大越"黑白分明"）。
- `p05 / p95`：5% 和 95% 分位亮度，反映动态范围。
- 这些是 fig02（尺寸）和 fig07（强度/对比度）的数据来源。

### 1.5 感知哈希 pHash（找"长得几乎一样"的近重复图）

```python
rec["phash"] = str(imagehash.phash(Image.open(ipath).convert("L")))
```

- pHash 把图缩小、做 DCT、取低频，生成一个 64 位指纹。**内容相似的图 → 指纹相近**（即使分辨率/压缩不同）。
- 两张图指纹的**汉明距离**（有多少位不同）越小越像：0=几乎同图，≤5=近重复。这是 fig09/fig10 的基础。

### 1.6 mask 并集 → 病灶像素、面积比

```python
union = np.zeros((H, W), np.uint8)
for mk in masks:
    mm = cv2.imread(..., cv2.IMREAD_GRAYSCALE)
    if mm.shape != (H, W):
        mm = cv2.resize(mm, (W, H), interpolation=cv2.INTER_NEAREST)  # 尺寸对齐
    union |= (mm > 127).astype(np.uint8)         # 二值化后按位或
lesion_px = int(union.sum())
rec["area_ratio"] = lesion_px / (W * H)          # 病灶占整图比例
rec["has_lesion"] = lesion_px > 0
```

- `mm > 127`：把 mask 二值化（白=1）。
- `union |= ...`：多个 mask 取**并集**（一图多病灶时合成一张），这正是二值分割训练时的做法。
- `INTER_NEAREST`：mask 缩放必须用最近邻，否则会插值出灰边、破坏二值。
- `area_ratio` 是 fig05 的核心；`has_lesion` 用来区分 normal（全黑 mask → False）。

### 1.7 形状特征（只对有病灶的图算）

先取最大外轮廓：

```python
cnts, _ = cv2.findContours(union, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
c = max(cnts, key=cv2.contourArea)               # 最大的病灶轮廓
area = cv2.contourArea(c); peri = cv2.arcLength(c, True)
hull = cv2.convexHull(c); harea = cv2.contourArea(hull)
bx, by, bw, bh = cv2.boundingRect(c)
```

然后算 4 个形状描述子（fig08 用）：

| 特征 | 公式 | 含义（越大代表） |
|---|---|---|
| **circularity 圆度** | `4π·area / peri²` | 越接近 1 越圆；越小越不规则/越有毛刺 |
| **solidity 密实度** | `area / 凸包面积` | 越接近 1 越"饱满"；有凹陷/分叶 → 变小 |
| **extent 占空比** | `area / (bw·bh)` | 病灶填满外接矩形的程度 |
| **eccentricity 偏心率** | `√(1 − b²/a²)`（拟合椭圆的长短轴 a,b） | 0=正圆，越接近 1 越"长条" |

```python
rec["circularity"] = 4*np.pi*area/(peri**2)
rec["solidity"]    = area/harea
rec["extent"]      = area/(bw*bh)
if len(c) >= 5:                                   # fitEllipse 至少需要 5 个点
    (_,_),(MA,ma),_ = cv2.fitEllipse(c)
    a_, b_ = max(MA,ma)/2, min(MA,ma)/2
    rec["eccentricity"] = np.sqrt(1 - b_**2/a_**2)
```

- 医学意义：良性病灶通常圆、光滑（圆度/solidity 高）；恶性常不规则、有毛刺（圆度/solidity 低、偏心率高）。fig08 正是验证这一点。

### 1.8 位置热力图累加（fig06 用）

```python
rmask = cv2.resize(union, (HM, HM), interpolation=cv2.INTER_NEAREST)  # 统一到 256×256
heatmaps[cls] += rmask                             # 同类所有病灶 mask 叠加
heat_counts[cls] += 1
```

- 所有图尺寸不同，先统一缩放到 256×256 再叠加，才能对齐坐标。
- 叠加后每个像素 = "有多少张图的病灶覆盖了这里"，除以张数就是**病灶出现频率**。存成 `heatmap_benign.npy` 等。

### 1.9 重复检测：精确 + 近似 + 聚类

**精确重复**（MD5 相同）：

```python
dup_exact = df.groupby("img_md5").size().reset_index(name="n")
exact_groups = dup_exact[dup_exact.n > 1]         # 出现 >1 次的指纹
```

**近重复**（pHash 汉明距离 ≤ 5）—— 两两比较 780 张：

```python
THRESH = 5
for i in range(len(fns)):
    for j in range(i+1, len(fns)):
        dist = hashes[fns[i]] - hashes[fns[j]]    # imagehash 的 "-" 就是汉明距离
        if dist <= THRESH:
            near_pairs.append((fns[i], fns[j], dist, cls_i, cls_j))
```

**并查集（union-find）把近重复对连成"簇"**：

```python
def find(x): ...          # 路径压缩
def union_(a,b): ...      # 合并两个集合
for a,b,*_ in near_pairs: union_(a,b)
clusters = {}             # 根 → 成员列表
```

- 为什么要聚类：A≈B、B≈C 时，A/B/C 应算作**同一个簇**。划分数据集时整个簇必须待在同一侧，否则泄漏。这是 fig10 的基础。

### 1.10 汇总 → summary.json

把所有关键统计（类别数、mask 数、尺寸范围、面积中位数、重复对数、跨类对数……）写进 `summary.json`，供画图和 PPT 直接引用（保证图上标的数字和表里一致）。

---

## 2. `02_make_figures.py`：10 张图逐张详解

统一约定：`COL = {benign: 绿, malignant: 红, normal: 灰}`（红=恶性=危险，直觉映射）。`save()` 负责 `tight_layout + 存 130dpi PNG`。

---

### fig01 — 类别分布

**画了什么**：左=柱状图（每类张数 + 百分比标签），右=饼图。

**代码怎么算**：
```python
cnt = df.cls.value_counts().reindex(CLS)          # 每类计数
ax[0].bar(CLS, cnt.values, ...)                   # 柱
ax[1].pie(cnt.values, autopct="%1.1f%%", ...)     # 饼
```

**怎么读 / 结论**：benign 437(56%) > malignant 210(27%) > normal 133(17%)，**类别不平衡**，良性约是正常的 3.3 倍 → 训练要用分层划分、class weights、平衡增强，否则模型偏向多数类。

---

### fig02 — 图像尺寸与宽高比

**画了什么**：左=宽 vs 高散点（按类别着色，虚线标 500×500 参考）；中=宽、高各自的直方图；右=宽高比 W/H 直方图（虚线标 1.0）。

**代码怎么算**：
```python
ax[0].scatter(s.width, s.height, color=COL[c])    # 每张图一个点
ax[1].hist(df.width); ax[1].hist(df.height)
ax[2].hist(df.aspect)                             # aspect = W/H
```

**怎么读 / 结论**：点云散得很开、并不聚在 500×500；标题写明 **639 种不同尺寸**（宽 190–1048，高 310–719）。说明数据集"平均 500×500"只是平均值，实际五花八门 → **必须统一 resize**；宽高比多数 >1（横图），建议 resize+padding 保形，别硬拉伸使病灶变形。

---

### fig03 — 样本可视化（图 + GT 轮廓）

**画了什么**：3 行（每类）× 4 列，随机抽样，灰度图上叠加病灶轮廓（正常无轮廓），标题标类别和尺寸。

**代码怎么算**：随机抽样后，对每张样本**重新读原图 + 重算 mask 并集**，用 `findContours` 提轮廓再 `ax.plot` 描线：
```python
sub = df[df.cls==c].sample(4)
img = np.asarray(Image.open(row.path).convert("L")); ax.imshow(img, cmap="gray")
# 重算 union（同 1.6）→ 找轮廓 → 画线
cnts,_ = cv2.findContours(union, cv2.RETR_EXTERNAL, ...)
for cc in cnts: ax.plot(cc[:,0,0], cc[:,0,1], color=COL[c], lw=2)
```

**怎么读 / 结论**：肉眼确认标注对齐正确；良性多为**椭圆、边界光滑**，恶性多为**不规则、边界模糊、有声影**；并且发现图上有**烧录的文字/测量 caliper**（如 "RT LOQ"、"RIGHT BREAST"）→ 有"捷径学习"风险，建议裁剪/inpaint 掉。

---

### fig04 — mask 可用性与图-mask 匹配

**画了什么**：左=有 mask / 无 mask 的图数；右="每张图有几个 mask 文件"的分布（纵轴对数）。

**代码怎么算**：
```python
ax[0].bar(["with mask","without mask"], [summary[...], summary[...]])
nm = df.n_masks.value_counts().sort_index()       # n_masks 的分布
ax[1].bar(nm.index, nm.values); ax[1].set_yscale("log")
```

**怎么读 / 结论**：780 张 **100% 都有 mask**（无孤儿图）；绝大多数是 1 个 mask，**17 张有多 mask**（16 良性 + 1 恶性，最多 3 个）→ 二值分割时对多 mask 取并集即可（1.6 已实现）。对数纵轴是因为多 mask 的数量太少，线性坐标看不见。

---

### fig05 — 病灶面积比

**画了什么**：左=良/恶两类的面积占比小提琴图（内含四分位线）；右=两类面积比的密度直方图。只用有病灶的图（normal 排除）。

**代码怎么算**：
```python
les = df[df.has_lesion]; les["area_pct"] = les.area_ratio*100
sns.violinplot(data=les[两类], x="cls", y="area_pct", inner="quartile")
ax[1].hist(..., density=True)
```

**怎么读 / 结论**：**恶性病灶明显更大**（面积中位数 12.1% vs 良性 3.8%）。含义有二：① 病灶大小本身是判别特征；② 很多良性病灶很小 → 像素级前景/背景极不平衡，Dice 对小目标敏感 → 建议 Dice+BCE 或 Tversky loss。

---

### fig06 — 病灶位置热力图

**画了什么**：三张热力图（benign/malignant/normal），颜色=该位置的病灶出现频率；青色十字标图像中心；normal 无病灶（全黑并标注）。

**代码怎么算**：读 1.8 存的累加图，除以该类张数得到频率：
```python
hm = np.load(f"heatmap_{c}.npy") / n              # n = 该类有病灶的图数
ax[i].imshow(hm, cmap="magma", extent=[0,1,1,0])  # 坐标归一化到 0~1
```

**怎么读 / 结论**：亮=病灶常出现的位置。两类都集中在**中间偏上**（医生把探头对准病灶居中放置 → 采集偏置）。含义：模型可能"靠位置作弊" → 用随机裁剪/平移增强提升鲁棒性。

---

### fig07 — 像素强度 / 对比度

**画了什么**：左=每张图平均亮度的 KDE 曲线（按类）；中=每张图对比度(std)的箱线图（按类）；右=聚合像素强度直方图（每类抽 30 张，把像素值堆起来）。

**代码怎么算**：
```python
sns.kdeplot(df[df.cls==c].intensity_mean)         # 左：逐图平均亮度分布
sns.boxplot(data=df, x="cls", y="intensity_std")  # 中：逐图对比度
# 右：抽样读像素，隔点取样(::20)防止太大，step 直方图
vals = np.concatenate([读图().ravel()[::20] for p in 抽样])
ax[2].hist(vals, histtype="step", density=True)
```

**怎么读 / 结论**：三类的亮度、对比度分布**高度重叠** → 光靠整体灰度很难分类，需要纹理/形状特征；不同图亮度差异大 → 训练前要**逐图归一化**（可选 CLAHE 增强低对比区域）。

---

### fig08 — 病灶形状分析

**画了什么**：4 个小提琴图并排（圆度 / solidity / extent / 偏心率），每个都比较良性 vs 恶性。

**代码怎么算**：直接用 1.7 算好的 4 列画小提琴：
```python
for f in ["circularity","solidity","extent","eccentricity"]:
    sns.violinplot(data=两类, x="cls", y=f, inner="quartile")
```

**怎么读 / 结论**：恶性的**圆度、solidity、extent 更低，偏心率更高** → 边界更不规则、更长条。定量印证了 fig03 的肉眼观察 → 支持用 Attention U-Net、边界感知 loss 抓不规则边界。

---

### fig09 — 重复 / 近重复检测

**画了什么**：左=每张图"到最近邻的最小汉明距离"直方图（虚线标 ≤1/≤5/≤10 三档）；右=近重复对示例拼图（第一对特意选**跨类**对）。

**代码怎么算**：
```python
# 把每个 phash 转成 64 位 0/1 向量，向量化算每张图到其它所有图的最小汉明距离
arr = 每张图的64位数组
for i: d = (arr[i] != arr).sum(1); d[i]=999; mind.append(d.min())
ax[0].hist(mind, bins=range(0,33))
# 右：从 near_duplicates.csv 里按距离排序，优先取一对跨类对，逐对贴原图
```

**怎么读 / 结论**：左图在**小距离处有明显一堆**（很多图存在近乎相同的另一张）。统计：186 对近重复(≤5)、涉及 277 张(35%)、124 个簇、其中 **10 对跨类**。右图最上面那对 **Ham=0 且一张标 benign 一张标 malignant**——同一张图被标了两个类（配合 MD5 检测，确认是**字节完全相同**的 `benign(433)` = `malignant(145)`）。这是 BUSI 已知缺陷 → 训练前必须去重 + 人工修标签。

---

### fig10 — 训练-测试泄漏检查

**画了什么**：左=近重复簇的大小分布（按大小排序的柱）；右=模拟随机划分时"被拆开的簇数"的分布（40 次模拟直方图，绿线在 0）。

**代码怎么算**：
```python
# 左：用 near_duplicates 重建并查集 → 每个簇成员数
sizes = sorted([len(v) for v in clusters if len(v)>1], reverse=True)
# 右：40 次随机 70/30 划分，统计有多少簇被拆到训练/测试两侧
for _ in range(40):
    随机划分 train/test
    for 每个簇:
        if 成员既在train又在test: leaked += 1
    leaks.append(leaked)
ax[1].hist(leaks)
```

**怎么读 / 结论**：随机 70/30 划分平均会把 **~59 个近重复簇**拆到训练/测试两侧 → 测试集里混进"训练见过的近乎同一张图" → **指标虚高**。对策：用**分组划分**（把同一簇、理想情况下同一患者，整体分到一侧），并先去重、修跨类冲突。这是整份 EDA 最重要的"落地建议"。

---

## 3. `03_build_ppt.py`（简述）

把上面 10 张图 + 每张的"关键要点"排进 16:9 幻灯片：

- `add_fig()`：按图片原始像素比例缩放，塞进版面框（图在左、要点在右）。
- `content_slide()`：标题 + 图 + 右侧 4 条 takeaway 的模板，10 张分析页都用它。
- 另加：标题页、数据概览页（直接读 `summary.json` 拼数字）、数据质量总结页、建模建议页。共 14 页 → `EDA/BUSI_EDA.pptx`。

---

## 附：几个概念小抄

- **pHash（感知哈希）**：把图变成 64 位指纹，内容相似 → 指纹相近。抗缩放/压缩，适合找近重复。
- **汉明距离**：两个等长 0/1 串有多少位不同。pHash 距离越小越像（0=几乎同图）。
- **RMS 对比度**：灰度像素的标准差，衡量图"黑白反差"大小。
- **圆度/solidity/extent/偏心率**：见 1.7 表格，都是刻画病灶"规不规则"的形状指标。
- **并查集(union-find)**：把"两两相似"的关系合并成"连通簇"的数据结构，用来把近重复图聚成组。
- **数据泄漏**：测试集里出现了和训练集近乎相同的样本，导致评估结果偏乐观、不可信。
