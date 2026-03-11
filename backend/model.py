# backend/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, k, s, p)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class Encoder(nn.Module):
    def __init__(self, base=32):
        super().__init__()
        c = base
        self.net = nn.Sequential(
            ConvBlock(6, c),
            ConvBlock(c, c),
            ConvBlock(c, 2*c, s=2),   # 1/2
            ConvBlock(2*c, 2*c),
            ConvBlock(2*c, 4*c, s=2), # 1/4
            ConvBlock(4*c, 4*c),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBlock(4*c, 2*c),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBlock(2*c, c),
            nn.Conv2d(c, 3, 1)
        )
    def forward(self, cover, secret_small):
        secret_up = F.interpolate(secret_small, size=cover.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([cover, secret_up], dim=1)
        residual = self.net(x)            
        
        stego = torch.clamp(cover + 0.02 * residual, 0.0, 1.0)
        return stego

class Decoder(nn.Module):
    def __init__(self, out_size=32, base=32):
        super().__init__()
        c = base
        self.backbone = nn.Sequential(
            ConvBlock(3, c), ConvBlock(c, c),
            ConvBlock(c, 2*c, s=2),
            ConvBlock(2*c, 2*c),
            ConvBlock(2*c, 4*c, s=2),
            ConvBlock(4*c, 4*c),
        )
        self.head = nn.Sequential(nn.Conv2d(4*c, 3, 1), nn.Sigmoid())
        self.out_size = out_size
    def forward(self, stego):
        x = self.backbone(stego)
        out = self.head(x)
        return F.interpolate(out, size=(self.out_size, self.out_size), mode='bilinear', align_corners=False)

class StegoSystem(nn.Module):
    def __init__(self, secret_size=32, base=32):
        super().__init__()
        self.encoder = Encoder(base=base)
        self.decoder = Decoder(out_size=secret_size, base=base)
    def forward(self, cover, secret):
        stego = self.encoder(cover, secret)
        rec = self.decoder(stego)
        return stego, rec