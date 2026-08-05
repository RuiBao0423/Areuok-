# -*- coding: utf-8 -*-
"""
Models/nnUNet_Isensee2021/model.py

Reproduction of the nnU-Net "blueprint" 2D U-Net (Isensee et al., Nature Methods
2021 / arXiv:1904.08128) for BUSI lesion segmentation.

We reproduce nnU-Net's *architecture blueprint* faithfully (this is the part that
is a fixed design, not the self-configuring heuristics):
  * 2 conv blocks per resolution stage, block order  Conv3x3 -> InstanceNorm -> LeakyReLU(0.01)
  * Instance Normalization (chosen by nnU-Net over BatchNorm for small batches)
  * LeakyReLU, negative slope 0.01
  * downsampling by STRIDED CONVOLUTION (no max-pool)
  * upsampling by TRANSPOSED CONVOLUTION
  * base 32 feature maps, doubled each downsample, capped at 512
  * DEEP SUPERVISION: auxiliary seg heads on the upper decoder resolutions,
    trained against downsampled ground truth (loss handled in train.py)

The self-configuring parts of nnU-Net (dataset fingerprinting, data-driven patch
size / spacing / batch size, 3D + cascade configs, 5-fold cross-val + ensembling,
sliding-window tiled inference) are OUT OF SCOPE and documented in README.md.

Interface parity with the seg baselines: input (N,1,256,256), output (N,1,256,256)
logits at full resolution. During training `forward(x, deep=True)` also returns the
auxiliary low-resolution logits for the deep-supervision loss.
"""
import torch
import torch.nn as nn


def _conv_block(cin, cout):
    """nnU-Net stage: 2 x (Conv3x3 -> InstanceNorm -> LeakyReLU(0.01))."""
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.InstanceNorm2d(cout, affine=True),
        nn.LeakyReLU(0.01, inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.InstanceNorm2d(cout, affine=True),
        nn.LeakyReLU(0.01, inplace=True),
    )


class NNUNet(nn.Module):
    """nnU-Net-blueprint 2D U-Net with deep supervision.

    features : per-stage channel widths (doubled, capped at 512 -> nnU-Net rule).
               5 stages on a 256 input -> resolutions 256/128/64/32/16.
    n_ds     : number of deep-supervision heads (upper decoder resolutions). nnU-Net
               supervises all but the two lowest resolutions.
    """
    def __init__(self, in_ch=1, n_classes=1, features=(32, 64, 128, 256, 512), n_ds=3):
        super().__init__()
        self.n_stages = len(features)
        self.n_ds = n_ds

        # ---- encoder: conv block per stage + strided-conv downsampler between stages
        self.enc = nn.ModuleList()
        self.down = nn.ModuleList()
        cin = in_ch
        for i, f in enumerate(features):
            self.enc.append(_conv_block(cin, f))
            if i < self.n_stages - 1:
                # strided conv downsample (nnU-Net uses conv stride 2, not pooling)
                self.down.append(nn.Conv2d(f, f, 3, stride=2, padding=1, bias=False))
            cin = f

        # ---- decoder: transposed-conv up + conv block on concatenated skip
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for i in range(self.n_stages - 1, 0, -1):
            self.up.append(nn.ConvTranspose2d(features[i], features[i - 1],
                                              kernel_size=2, stride=2))
            self.dec.append(_conv_block(features[i - 1] * 2, features[i - 1]))

        # ---- segmentation heads (1x1 conv). Head 0 = full resolution (main output);
        #      heads 1..n_ds = deep-supervision on the next-lower decoder resolutions.
        self.heads = nn.ModuleList(
            [nn.Conv2d(features[i], n_classes, 1) for i in range(n_ds + 1)]
        )

    def forward(self, x, deep=False):
        # encoder
        skips = []
        for i in range(self.n_stages):
            x = self.enc[i](x)
            if i < self.n_stages - 1:
                skips.append(x)
                x = self.down[i](x)
        # decoder (collect outputs from full-res downwards for deep supervision)
        dec_outs = []
        for j in range(self.n_stages - 1):
            x = self.up[j](x)
            skip = skips[self.n_stages - 2 - j]
            x = torch.cat([x, skip], dim=1)
            x = self.dec[j](x)
            dec_outs.append(x)                       # resolutions: .../ ... /256 (last)
        dec_outs = dec_outs[::-1]                     # now [full-res, half, quarter, ...]

        main = self.heads[0](dec_outs[0])
        if not deep:
            return main
        aux = [self.heads[k](dec_outs[k]) for k in range(1, self.n_ds + 1)]
        return [main] + aux                           # highest-res first


def count_params_m(model):
    return round(sum(p.numel() for p in model.parameters()) / 1e6, 3)
