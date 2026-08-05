# -*- coding: utf-8 -*-
"""
U-Net (Ronneberger, Fischer & Brox, MICCAI 2015)  ---  our from-scratch PyTorch
implementation of the architecture described in the paper (Fig. 1):
a contracting (encoder) path + symmetric expanding (decoder) path with skip
connections. Deviations from the 2015 paper, made for this small BUSI task, are
noted in README.md (padded 3x3 convs so input/output sizes match; BatchNorm;
sigmoid output for 1-class lesion-vs-background).
"""
import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """(conv 3x3 -> BN -> ReLU) x2, as the two 3x3 convs in each U-Net block."""
    def __init__(self, cin, cout):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base=64):
        super().__init__()
        c = [base, base*2, base*4, base*8, base*16]     # 64,128,256,512,1024
        # encoder
        self.enc1 = DoubleConv(in_ch, c[0])
        self.enc2 = DoubleConv(c[0], c[1])
        self.enc3 = DoubleConv(c[1], c[2])
        self.enc4 = DoubleConv(c[2], c[3])
        self.pool = nn.MaxPool2d(2)
        self.bott = DoubleConv(c[3], c[4])
        # decoder (up-conv 2x2 that halves channels, then concat skip, then DoubleConv)
        self.up4 = nn.ConvTranspose2d(c[4], c[3], 2, stride=2); self.dec4 = DoubleConv(c[4], c[3])
        self.up3 = nn.ConvTranspose2d(c[3], c[2], 2, stride=2); self.dec3 = DoubleConv(c[3], c[2])
        self.up2 = nn.ConvTranspose2d(c[2], c[1], 2, stride=2); self.dec2 = DoubleConv(c[2], c[1])
        self.up1 = nn.ConvTranspose2d(c[1], c[0], 2, stride=2); self.dec1 = DoubleConv(c[1], c[0])
        self.head = nn.Conv2d(c[0], out_ch, 1)           # 1x1 conv -> class map

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bott(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b),  e4], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.head(d1)                              # logits (B,1,H,W)


if __name__ == "__main__":
    m = UNet()
    x = torch.randn(2, 1, 256, 256)
    print("output:", m(x).shape, "| params:", sum(p.numel() for p in m.parameters())/1e6, "M")
