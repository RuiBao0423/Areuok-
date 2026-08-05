# FCN-AlexNet (transfer learning) [02] — source & attribution

**Paper reproduced:** Yap et al., *Automated Breast Ultrasound Lesions Detection Using
Convolutional Neural Networks*, IEEE J. Biomedical and Health Informatics, 22(4), 2018.
DOI 10.1109/JBHI.2017.2731873. This is the paper's **best** method (Section IV-A-3,
"Transfer Learning"): a **fully convolutional network based on a pretrained AlexNet**
(FCN-AlexNet), which "out-performed other methods for lesion detection" (their Table I:
F-measure 0.91 / 0.89 on Datasets A / B). The FCN idea comes from Long, Shelhamer &
Darrell, *Fully Convolutional Networks for Semantic Segmentation*, CVPR 2015.

## What is **our own work**
- `model.py` — we take torchvision's **ImageNet-pretrained AlexNet** convolutional
  backbone (the "transfer learning") and add our own **fully-convolutional head**
  (fc6/fc7 re-cast as conv layers + a 1×1 score conv) with bilinear upsampling to the
  input resolution. Written from scratch in PyTorch.
- `train.py` — the full fine-tuning + evaluation pipeline we wrote (SGD as in the paper,
  Dice+BCE loss, test inference, seg **and** detection metrics, figures/JSON export).
- `../common/data.py`, `../common/metrics.py` — shared cleaning, grouped split, 3-channel
  ImageNet-normalised inputs, and metrics.

## Faithful to the paper
- ImageNet-pretrained AlexNet made fully convolutional (transfer learning);
- **SGD, learning rate 0.001, 60 epochs, dropout 0.33** (their Section IV-C);
- evaluated with the paper's detection metrics (TPF / FPs-per-image / F-measure).

## Deviations (deliberate)
| Paper (2018) | Ours | Why |
|---|---|---|
| Caffe FCN-AlexNet, fc6/fc7 convolutionalised from pretrained weights | PyTorch; pretrained AlexNet **backbone** + fresh conv head (kernel-3, keeps spatial) | torchvision has no FCN-AlexNet; our head keeps a finer grid so Dice is also meaningful |
| Datasets A/B, detection-only, 10-fold CV | BUSI benign+malignant (645), single grouped split, + Dice/IoU | our dataset; leakage-free split from EDA; adds seg metrics for U-Net comparison |
| 3-channel handling in Caffe | grayscale replicated to 3 channels + ImageNet normalisation | BUSI is pseudo-RGB grayscale; AlexNet expects 3×ImageNet inputs |

Run: `python Models/Yap2018_FCN_AlexNet/train.py` → writes `Models/results/Yap2018_FCN_AlexNet.json`.
