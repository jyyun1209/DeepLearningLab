import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

## Model part
class Conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()
        self.down1 = Conv(in_channels, 64)
        self.down2 = Conv(64, 128)
        self.down3 = Conv(128, 256)
        self.down4 = Conv(256, 512)
        self.down5 = Conv(512, 1024)
        self.maxpool2 = nn.MaxPool2d(2)
        self.maxpool3 = nn.MaxPool2d(3)

    def forward(self, x):
        x = self.down1(x)
        x = self.maxpool2(x)
        x = self.down2(x)
        x = self.maxpool2(x)
        x = self.down3(x)
        x = self.maxpool2(x)
        x = self.down4(x)
        x = self.maxpool2(x)
        x = self.down5(x)
        x = self.maxpool3(x)

        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.up1 = Conv(in_channels, 512)
        self.up2 = Conv(512, 256)
        self.up3 = Conv(256, 128)
        self.up4 = Conv(128, 64)
        self.up5 = Conv(64, out_channels)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample3 = nn.Upsample(scale_factor=3, mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.up1(x)
        x = self.upsample3(x)
        x = self.up2(x)
        x = self.upsample2(x)
        x = self.up3(x)
        x = self.upsample2(x)
        x = self.up4(x)
        x = self.upsample2(x)
        x = self.up5(x)
        x = self.upsample2(x)

        return x


class Simple_seg(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(Simple_seg, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.in_conv = Conv(n_channels, 64)
        self.encoder = Encoder(64, 1024)
        self.decoder = Decoder(1024, 64)
        self.out_conv = Conv(64, n_classes)

    def forward(self, x):
        x = self.in_conv(x)
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.out_conv(x)

        return x