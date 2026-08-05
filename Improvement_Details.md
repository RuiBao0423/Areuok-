# Improvement 部分详解 — 分割器 & 双阶段 pipeline

> COMP9444 Project 047 · BUSI 乳腺超声。本文件汇总我们**两个自研改进模型**的做法、优化细节、评估方式与**消融实验**。所有数字取自 `Models/results/*.json` 与报告表格，测试集固定为无泄漏 grouped split 的 **98 张 benign+malignant** 图。

---

## 0. 总览：我们做了两个改进

| 改进 | 基于什么 | 一句话 | 最好结果 |
|---|---|---|---|
| **① 改进分割器** Improved Segmenter | 从零 U-Net [3] | 把 encoder 换成 ImageNet 预训练的 **EfficientNet-B4** | **Dice 0.760**（全场最高）|
| **② 改进双阶段** Improved Dual-stage | Bruno [17] | 换更强的 Stage-1 + 强化 Stage-2 + **严格评估(OOF/CI)** | **macro-F1 0.875**（2-class）|

核心方法论：**每加一个组件都做受控消融，用数字证明它到底有没有用**，而不是把技巧一股脑堆上去然后报一个好看的单次分数。

---

## 1. 改进分割器 (Improved Segmenter)

### 1.1 动机
在跑基线时发现一条规律：**唯一用了迁移学习的分割器（DeepLabV3+，预训练 ResNet-50 encoder）明显打赢所有从零训练的模型**。
→ 假设：**在 BUSI 这种小数据（训练集仅 445 张）上，迁移学习才是决定性因素**。于是从"给 U-Net 换预训练 encoder"入手。

### 1.2 怎么做（三步，逐步叠加）

1. **换 encoder（关键一步）**
   保留 U-Net 的 **decoder + skip 连接结构不变**，只把 encoder 整块替换成 **ImageNet 预训练的 EfficientNet-B4**（用 `segmentation-models-pytorch` 现成实现）。
   - 因为预训练 encoder 是按 3 通道 ImageNet 训练的 → 输入把灰度图复制成伪 RGB、用 ImageNet 均值/方差归一化。

2. **换损失：Focal Tversky + BCE**
   Focal Tversky 对 **false negative（漏检）加重惩罚**，用来打击"完全漏掉病灶"的长尾（基线的 Dice 标准差很大，说明有一批病灶被彻底漏掉）。

3. **加推理技巧：TTA + 后处理**
   Test-Time Augmentation（水平翻转 + 多尺度）取平均 + **最大连通域(largest-CC)后处理**去掉零碎误检。

### 1.3 具体配置

| 项 | 从零 U-Net（基线）| 改进分割器 |
|---|---|---|
| 架构 | U-Net（encoder 从零）| U-Net decoder + **EffB4 预训练 encoder** (smp) |
| 输入 | 256×256 灰度 | 256×256 灰度→伪 RGB + ImageNet 归一化 |
| 损失 | Dice + BCE | **Focal Tversky + BCE** |
| 优化器 | Adam, lr 1e-3 | **AdamW, lr 1e-3, cosine 调度** |
| epochs / batch | 60 / 8 | 60 / 8 |
| 推理 | 直接输出 | **TTA(hflip+multiscale) + largest-CC** |
| 参数量 | 31.0M | **20.2M（更少）** |
| 训练耗时 | ~58 min | **~8 min（预训练收敛快约 7 倍）** |

> 亮点：改进模型**参数更少、训练更快、分数还更高**——预训练迁移的三重红利。

### 1.4 评估方式（分割怎么测）
- 测试集：无泄漏 grouped split 的 **98 张** benign+malignant 图（有病灶才评分割）。
- 阈值固定 **0.5**。
- 指标：**逐图算 Dice / IoU 再平均**（不把所有像素堆一起，否则大病灶主导）；辅以 **像素 precision/recall**、pixel accuracy。
- 额外看 **Dice 标准差**：衡量"完全漏检"的长尾有多严重。

### 1.5 消融实验（每行加一个组件，报测试 Dice）

| 配置 | Dice | Dice std | 像素 recall |
|---|---|---|---|
| 从零 U-Net（Dice+BCE）| 0.648 | 0.362 | 0.637 |
| **+ 预训练 EffB4 encoder** | **0.765** | **0.285** | 0.763 |
| + Focal Tversky 损失 | 0.760 | 0.282 | **0.800** |
| + TTA + 后处理（完整模型）| 0.760 | 0.307 | 0.790 |

**结论（一句话）：预训练 encoder 一项就贡献了几乎全部提升（+0.117 Dice），并把 Dice 标准差 0.362→0.285（漏检的长尾大幅减少）。** Focal Tversky 不涨 Dice，只是把"精度↔召回"往"少漏检"那边挪（像素 recall 0.763→0.800）；TTA/后处理在本数据上无 Dice 增益。
→ **迁移学习 > 架构/损失/推理上的花活。**

---

## 2. 改进双阶段 pipeline (Improved Dual-stage)

### 2.1 为什么选 dual-stage（而不是 MTL-OCA）
两个联合模型里，**MTL-OCA[25] 共享一个 backbone，在两个任务上都吃亏**（seg 0.612 / cls 0.720）。dual-stage 的优势：
- **模块化**：Stage-1 分割器、Stage-2 分类器可各自独立替换/改进。
- **可做受控实验**：固定 Stage-2、只换 Stage-1 → 单独测出分割器的真实贡献（联合模型做不到）。
- **复用最强专家模型** + **输出显式 mask+ROI，临床可解释**。

### 2.2 pipeline 结构
```
输入超声图 → [Stage-1 分割器] → 预测掩膜 → [ROI 提取: 最大连通域 + bbox外扩15% 裁剪]
           → ROI crop → [Stage-2 分类器 EfficientNet-B0] → benign / malignant
```
沿用 Bruno[17] 设定：**只做 benign/malignant 二分类**、灰度 256×256、找不到病灶就退回整张图裁 ROI。

### 2.3 我们的优化细节（三层，逐层加）

1. **Stage-1 换更强的分割器（受控替换）**
   把 Bruno 的 DeepLabV3+ 换成我们的**改进分割器**（EffB4）。**Stage-2 与 ROI 耦合逐字节保持不变** → pipeline 分数的任何变化都只归因于分割。
   - 结果：Stage-1 Dice 0.739→0.760，但 pipeline F1 只 0.804→0.807（→ 引出"瓶颈在分类器"）。

2. **强化 Stage-2（StrongCls）**
   原 pipeline 有个隐患：Stage-2 在**干净的 GT 掩膜裁剪**上训练，却在**不完美的预测掩膜裁剪**上测试（train/test 分布不匹配）。
   修法：让 Stage-2 在 **GT 掩膜 + 预测掩膜两种 ROI crop 上都训练**（2 倍数据、分布对齐），60 epochs cosine，验证集用预测 ROI 选模型，加水平翻转 TTA。
   - 结果：F1 0.807→0.817。

3. **融合手工特征（尝试）**
   从预测掩膜算 **shape 特征(6 个)** 或 **radiomics(14 个 = 形状 + GLCM 纹理)**，拼进分类器。
   - 单次看似有用（+shape 单次 0.840），但严格评估下站不住（见 2.5）。

### 2.4 严格评估方式（这才是双阶段的核心卖点）
为避免"单次跑分虚高、噪声当成果"，我们用四件套：

- **OOF 交叉拟合(out-of-fold, 5-fold)**：Stage-2 要在"预测掩膜裁剪"上训练，就需要训练图的预测掩膜；但不能用见过这些图的分割器（掩膜会假性完美）。→ 用 5 折交叉拟合，**每张图的掩膜由只在其他 4 折上训练的分割器产生**。这是**正确性修正**，不是刷分。
- **5-seed 集成**：每个配置跑 5 个随机种子，对 softmax 取平均，消除单次运气。
- **bootstrap 95% CI**：对 98 张测试集有放回重抽 → 每次重算 macro-F1 → 取 2.5%/97.5% 分位得置信区间，判断差异是不是噪声。
- **GT-ROI 上限**：把 Stage-2 喂**真值掩膜裁剪**，得到"完美分割"的性能天花板（~0.87），用来定位误差到底出在哪。

### 2.5 消融实验

**(a) 单次阶梯（single-run，仅供参考——看着像稳步上升）**

| pipeline | Stage-1 Dice | pipeline F1 | Acc | GT-ROI F1(上限) |
|---|---|---|---|---|
| Bruno (DeepLabV3+ → EffB0) | 0.739 | 0.804 | 0.827 | 0.865 |
| Ours: 换 Improved-UNet Stage-1 | 0.760 | 0.807 | 0.827 | 0.884 |
| Ours: + 强化 Stage-2 | 0.760 | 0.817 | 0.837 | 0.882 |
| Ours: + shape 特征 | 0.761 | 0.840 | 0.857 | 0.850 |

**(b) 严格评估（OOF + 5-seed 集成 + bootstrap CI——诚实结论）**

| 配置（全 OOF）| per-seed F1 (mean±std) | 5-seed 集成 F1 | 95% CI |
|---|---|---|---|
| 强化 Stage-2（纯 CNN，无手工特征）| 0.847 ± 0.011 | **0.875** | [0.80, 0.94] |
| + shape (6 特征) | 0.861 ± 0.016 | **0.875** | [0.80, 0.94] |
| + radiomics (14 特征) | 0.862 ± 0.013 | 0.863 | [0.79, 0.93] |

**两条结论：**
1. **瓶颈在分类器，不在分割器**：Stage-1 Dice 0.739→0.760，pipeline F1 几乎不动(0.804→0.807)；即便喂完美掩膜也只到 ~0.87 上限，远高于实际分数 → 剩下的误差是分类器自身的，不是掩膜不好。**掩膜够糙也够用来裁 ROI 了。**
2. **只有集成是真的**：(a) 的"漂亮阶梯"在 (b) 的 CI 下几乎完全重叠——**手工特征(shape/radiomics)无可靠增益**；唯一稳健的提升是**集成**（per-seed 0.847 → 集成 0.875）。在 98 张测试集上，单次 ±0.05 以内的差距统计上分不清。

---

## 3. 数据如何测量（评估协议汇总）

| 维度 | 分割（segmenter）| 双阶段（dual-stage）|
|---|---|---|
| 测试集 | 98 张 benign+malignant | 98 张 benign+malignant（2-class）|
| 主指标 | **逐图 Dice / IoU**（阈值 0.5）| **macro-F1** |
| 次指标 | 像素 P/R、Dice std | balanced acc、accuracy、macro-AUC、per-class P/R/F1、混淆矩阵 |
| 稳健性 | — | **OOF 交叉拟合 + 5-seed 集成 + bootstrap 95% CI** |
| 参照/上限 | 论文报告值（对标）| **GT-ROI 完美分割上限** |
| 划分 | 无泄漏 grouped 70/15/15，所有模型同一划分 | 同左 |
| 调参 | 只用 train/val | 只用 train/val |

---

## 4. 一页话总结
- **分割器**：换 **ImageNet 预训练 EffB4 encoder** 是决定性一步（Dice 0.648→0.765，还更快更省参数）；损失/TTA 基本不涨。→ **迁移学习 > 复杂架构**。
- **双阶段**：**瓶颈在分类器**（换更强分割器几乎不动分数，GT-ROI 上限远在其上）；严格评估下**手工特征无用、只有集成稳健**，最佳 macro-F1 **0.875 (CI [0.80,0.94])**。
- **方法论主线**：**干净的评估（去泄漏 + OOF + CI）+ 迁移学习**，胜过架构上的花活——在小而噪声大的超声数据上尤其如此。
