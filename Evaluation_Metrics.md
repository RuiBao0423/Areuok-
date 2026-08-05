# Evaluation Metrics — baseline 要测的指标

> BUSI 项目 047。分类 = normal / benign / malignant（3 类，失衡）；分割 = 病灶 vs 背景（只在 benign+malignant 的 647 张上评）。
> ⚠️ 所有指标都要在 **分组/患者级划分（grouped split）** 的测试集上算，否则近重复泄漏会让分数虚高。

---

## 🔹 分类任务 (Classification)

| 指标 | 主/次 | 为什么 | 计算 |
|---|---|---|---|
| **Macro-F1** | ⭐主 | 失衡下的主指标，对小类公平 | `f1_score(y,ŷ,average='macro')` |
| **Per-class Precision / Recall / F1** | ⭐主 | 看每类，尤其 **malignant 的 Recall**（漏诊恶性代价最大） | `classification_report(y,ŷ)` |
| **ROC-AUC (macro, one-vs-rest)** | ⭐主 | 与阈值无关，失衡下比 accuracy 稳健 | `roc_auc_score(y,proba,multi_class='ovr',average='macro')` |
| **Confusion Matrix** | ⭐必看 | 看清 benign↔malignant 混淆 | `confusion_matrix(y,ŷ)` |
| Accuracy | 次 | 会看但别只看（失衡会虚高） | `accuracy_score(y,ŷ)` |
| Balanced Accuracy | 次 | accuracy 的失衡修正版 | `balanced_accuracy_score(y,ŷ)` |

**报告方式**：主看 **Macro-F1 + malignant Recall + macro-AUC**，附混淆矩阵。

---

## 🔹 分割任务 (Segmentation)

| 指标 | 主/次 | 为什么 | 计算 |
|---|---|---|---|
| **Dice / DSC** | ⭐主 | 医学分割标准指标；对**小病灶敏感**（benign 中位仅 3.8%） | 见下方代码 |
| **IoU / Jaccard** | ⭐主 | 与 Dice 互补，一起报 | 见下方代码 |
| **Pixel Precision / Recall（=Sensitivity）** | 次 | 看是漏检还是过分割 | 像素级 TP/FP/FN |
| Pixel Accuracy | 次 | 辅助（背景占多会虚高，别单看） | `(pred==gt).mean()` |
| (可选) Hausdorff HD95 | 加分 | 边界质量/最差偏差 | `scipy`/`medpy` |

**报告方式**：主看 **mean Dice + mean IoU**（对整个测试集逐图算再平均）。阈值固定 0.5。

---

## 🔧 通用规则
- **划分**：grouped/patient-aware split（先去重、修跨类冲突）→ 指标才可信。
- **稳健性**：多个 seed / K-fold，报告 **mean ± std**，不要只跑一次。
- **平均方式**：分类用 **macro**（对失衡公平）；分割 **逐图算 Dice 再平均**（不要把所有像素堆一起算，会被大病灶主导）。
- **对标数值**（论文报的，供参考不照搬）：分类 [17] AUC≈0.99；分割 [13] Dice 82.9%、[17] Dice 0.77、[16] Dice≈0.61。

---

## 📋 copy-paste：核心指标函数

```python
import numpy as np
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, roc_auc_score, balanced_accuracy_score)

# ---------- 分类 ----------
def eval_classification(y_true, y_pred, y_proba, labels=("normal","benign","malignant")):
    print(classification_report(y_true, y_pred, target_names=labels, digits=4))
    print("Macro-F1     :", round(f1_score(y_true, y_pred, average="macro"), 4))
    print("Balanced Acc :", round(balanced_accuracy_score(y_true, y_pred), 4))
    print("Macro AUC    :", round(roc_auc_score(y_true, y_proba,
                                    multi_class="ovr", average="macro"), 4))
    print("Confusion:\n", confusion_matrix(y_true, y_pred))

# ---------- 分割（逐图，pred/gt 是 0/1 掩膜） ----------
def dice(pred, gt, eps=1e-6):
    inter = (pred & gt).sum()
    return (2*inter + eps) / (pred.sum() + gt.sum() + eps)

def iou(pred, gt, eps=1e-6):
    inter = (pred & gt).sum()
    union = (pred | gt).sum()
    return (inter + eps) / (union + eps)

def eval_segmentation(preds, gts):        # preds/gts: list of 0/1 masks
    ds = [dice(p.astype(bool), g.astype(bool)) for p, g in zip(preds, gts)]
    js = [iou(p.astype(bool),  g.astype(bool)) for p, g in zip(preds, gts)]
    print(f"mean Dice = {np.mean(ds):.4f} ± {np.std(ds):.4f}")
    print(f"mean IoU  = {np.mean(js):.4f} ± {np.std(js):.4f}")
    return np.mean(ds), np.mean(js)
```

> 需要 `torchmetrics` 版本（GPU、训练中在线监控）的话告诉我，我再给一份。
