# -*- coding: utf-8 -*-
"""
Attention U-Net (Oktay et al., 2018, arXiv:1804.03999 -- "Attention U-Net:
Learning Where to Look for the Pancreas") for BUSI binary lesion segmentation.
Only the NETWORK is defined here; training lives in train.py.

Architecture (paper, adapted to this small 256x256 grayscale task):
  A standard U-Net encoder-decoder, but every skip connection passes through an
  ADDITIVE ATTENTION GATE (AG) before it is concatenated in the decoder. The gate
  uses the coarser decoder feature (the "gating signal" g, which carries more
  semantic/where-is-the-object context) to compute a soft spatial attention map
  alpha in [0,1], and multiplies the encoder skip feature by alpha. This suppresses
  responses in irrelevant background regions and highlights salient lesion regions
  -- exactly the "learning where to look" idea of the paper (Fig. 1 & Eq. 1-2).

Attention gate (Oktay Eq. 1):
    q = ReLU( W_x * x  +  W_g * g )          # additive attention
    alpha = sigmoid( psi * q )               # 1-channel spatial gate in [0,1]
    x_hat = alpha * x                        # re-weighted skip feature
(x and g are 1x1-convolved to a shared intermediate channel count F_int; g is the
gating signal from one level below, upsampled to x's resolution.)

I/O (matches common/data.py SegDataset with rgb3=False -- the from-scratch setting):
  input : (B, 1, 256, 256)  grayscale normalised to [-1, 1]
  output: (B, 1, 256, 256)  raw logits  -> use with BCEWithLogitsLoss
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(conv 3x3 -> BN -> ReLU) x2, the two 3x3 convs in each U-Net block."""
    def __init__(self, cin, cout):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class AttentionGate(nn.Module):
    """Additive attention gate (Oktay et al. 2018, Eq. 1-2).

    x : encoder skip feature      (B, F_l, H, W)
    g : gating signal (decoder)   (B, F_g, H', W')  -- from one level below
    Returns x re-weighted by a learned spatial attention map alpha in [0,1].
    """
    def __init__(self, f_x, f_g, f_int):
        super().__init__()
        self.theta_x = nn.Conv2d(f_x, f_int, 1, bias=False)   # W_x
        self.phi_g   = nn.Conv2d(f_g, f_int, 1, bias=True)     # W_g
        self.psi     = nn.Conv2d(f_int, 1, 1, bias=True)       # psi -> 1 channel
        self.relu    = nn.ReLU(inplace=True)

    def forward(self, x, g):
        if g.shape[-2:] != x.shape[-2:]:                       # align gating signal to skip
            g = F.interpolate(g, size=x.shape[-2:], mode="bilinear", align_corners=False)
        q = self.relu(self.theta_x(x) + self.phi_g(g))         # additive attention
        alpha = torch.sigmoid(self.psi(q))                     # (B,1,H,W) in [0,1]
        return x * alpha                                       # gated skip feature


class AttentionUNet(nn.Module):
    """U-Net with attention-gated skip connections (Oktay et al., 2018)."""
    def __init__(self, in_ch=1, out_ch=1, base=64):
        super().__init__()
        c = [base, base*2, base*4, base*8, base*16]            # 64,128,256,512,1024
        # encoder
        self.enc1 = DoubleConv(in_ch, c[0])
        self.enc2 = DoubleConv(c[0], c[1])
        self.enc3 = DoubleConv(c[1], c[2])
        self.enc4 = DoubleConv(c[2], c[3])
        self.pool = nn.MaxPool2d(2)
        self.bott = DoubleConv(c[3], c[4])
        # decoder: up-conv (halves channels) -> attention-gate the skip -> concat -> DoubleConv
        self.up4 = nn.ConvTranspose2d(c[4], c[3], 2, stride=2)
        self.att4 = AttentionGate(f_x=c[3], f_g=c[3], f_int=c[3]//2)
        self.dec4 = DoubleConv(c[4], c[3])
        self.up3 = nn.ConvTranspose2d(c[3], c[2], 2, stride=2)
        self.att3 = AttentionGate(f_x=c[2], f_g=c[2], f_int=c[2]//2)
        self.dec3 = DoubleConv(c[3], c[2])
        self.up2 = nn.ConvTranspose2d(c[2], c[1], 2, stride=2)
        self.att2 = AttentionGate(f_x=c[1], f_g=c[1], f_int=c[1]//2)
        self.dec2 = DoubleConv(c[2], c[1])
        self.up1 = nn.ConvTranspose2d(c[1], c[0], 2, stride=2)
        self.att1 = AttentionGate(f_x=c[0], f_g=c[0], f_int=c[0]//2)
        self.dec1 = DoubleConv(c[1], c[0])
        self.head = nn.Conv2d(c[0], out_ch, 1)                 # 1x1 conv -> class map

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bott(self.pool(e4))
        # each decoder step gates its skip with the just-upsampled decoder feature
        u4 = self.up4(b);  d4 = self.dec4(torch.cat([u4, self.att4(e4, u4)], 1))
        u3 = self.up3(d4); d3 = self.dec3(torch.cat([u3, self.att3(e3, u3)], 1))
        u2 = self.up2(d3); d2 = self.dec2(torch.cat([u2, self.att2(e2, u2)], 1))
        u1 = self.up1(d2); d1 = self.dec1(torch.cat([u1, self.att1(e1, u1)], 1))
        return self.head(d1)                                   # logits (B,1,H,W)


if __name__ == "__main__":
    m = AttentionUNet()
    x = torch.randn(2, 1, 256, 256)
    y = m(x)
    print("output:", tuple(y.shape), "| params: %.2fM" % (sum(p.numel() for p in m.parameters())/1e6))
    assert tuple(y.shape) == (2, 1, 256, 256)
