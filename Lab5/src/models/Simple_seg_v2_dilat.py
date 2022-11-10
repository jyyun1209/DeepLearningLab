import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

## Model part
class Conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class Simple_seg_v2_dilat(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(Simple_seg_v2_dilat, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.in_conv = Conv(n_channels, 64)
        self.down1 = Conv(64, 64)
        self.down2 = Conv(64, 128)
        self.down3 = Conv(128, 256)
        self.down4 = Conv(256, 512)
        self.down5 = Conv(512, 1024)
        self.maxpool2 = nn.MaxPool2d(2)
        self.maxpool3 = nn.MaxPool2d(3)
        self.up11 = Conv(1024, 512)
        self.up12 = Conv(1024, 512)
        self.up21 = Conv(512, 256)
        self.up22 = Conv(512, 256)
        self.up31 = Conv(256, 128)
        self.up32 = Conv(256, 128)
        self.up41 = Conv(128, 64)
        self.up42 = Conv(128, 64)
        self.up51 = Conv(64, 64)
        self.up52 = Conv(128, 64)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample3 = nn.Upsample(scale_factor=3, mode='bilinear', align_corners=True)
        self.out_conv = Conv(64, n_classes)

    def forward(self, x):
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x2 = self.maxpool2(x2)
        x3 = self.down2(x2)
        x3 = self.maxpool2(x3)
        x4 = self.down3(x3)
        x4 = self.maxpool2(x4)
        x5 = self.down4(x4)
        x5 = self.maxpool2(x5)
        x6 = self.down5(x5)
        x6 = self.maxpool3(x6)

        x = self.up11(x6)
        x = self.upsample3(x)
        x = torch.cat([x, x5], dim=1)
        x = self.up12(x)
        x = self.up21(x)
        x = self.upsample2(x)
        x = torch.cat([x, x4], dim=1)
        x = self.up22(x)
        x = self.up31(x)
        x = self.upsample2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.up32(x)
        x = self.up41(x)
        x = self.upsample2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.up42(x)
        x = self.up51(x)
        x = self.upsample2(x)
        x = torch.cat([x, x1], dim=1)
        x = self.up52(x)

        x = self.out_conv(x)

        return x