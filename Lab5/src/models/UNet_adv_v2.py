import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# torch.set_printoptions(profile="full")

## Model part
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, dilation=1):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, dilation=dilation),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, dilation=1):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, dilation=dilation)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet_adv_v2(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet_adv_v2, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc_d1 = DoubleConv(n_channels, 64, dilation=1)
        self.inc_d2 = DoubleConv(n_channels, 64, dilation=2)
        self.inc_d3 = DoubleConv(n_channels, 64, dilation=3)
        self.inc_d4 = DoubleConv(n_channels, 64, dilation=4)
        self.inc_d5 = DoubleConv(n_channels, 64, dilation=5)
        self.down1_d1 = Down(64, 128, dilation=1)
        self.down1_d2 = Down(64, 128, dilation=2)
        self.down1_d3 = Down(64, 128, dilation=3)
        self.down1_d4 = Down(64, 128, dilation=4)
        self.down1_d5 = Down(64, 128, dilation=5)
        self.down2_d1 = Down(128, 256, dilation=1)
        self.down2_d2 = Down(128, 256, dilation=2)
        self.down2_d3 = Down(128, 256, dilation=3)
        self.down2_d4 = Down(128, 256, dilation=4)
        self.down2_d5 = Down(128, 256, dilation=5)
        self.down3_d1 = Down(256, 512, dilation=1)
        self.down3_d2 = Down(256, 512, dilation=2)
        self.down3_d3 = Down(256, 512, dilation=3)
        self.down3_d4 = Down(256, 512, dilation=4)
        self.down3_d5 = Down(256, 512, dilation=5)
        factor = 2 if bilinear else 1
        self.down4_d1 = Down(512, 1024 // factor, dilation=1)
        self.down4_d2 = Down(512, 1024 // factor, dilation=2)
        self.down4_d3 = Down(512, 1024 // factor, dilation=3)
        self.down4_d4 = Down(512, 1024 // factor, dilation=4)
        self.down4_d5 = Down(512, 1024 // factor, dilation=5)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        self.out_up = nn.Upsample(
                scale_factor=240/236, mode='bilinear', align_corners=True)

    def forward(self, x):
        ###### fill the codes below #####
        # process of U-Net using predefined modules.
        # x1_d5 = self.inc_d5(x)

        # x1_d4 = self.inc_d4(x)
        # x2_d4 = self.down1_d4(x1_d4)

        # x1_d3 = self.inc_d3(x)
        # x2_d3 = self.down1_d3(x1_d3)
        # x3_d3 = self.down2_d3(x2_d3)

        # x1_d2 = self.inc_d2(x)
        # x2_d2 = self.down1_d2(x1_d2)
        # x3_d2 = self.down2_d2(x2_d2)
        # x4_d2 = self.down3_d2(x3_d2)

        x1_d1 = self.inc_d1(x)
        x2_d1 = self.down1_d1(x1_d1)
        x3_d1 = self.down2_d1(x2_d1)
        x4_d1 = self.down3_d1(x3_d1)
        x5_d1 = self.down4_d1(x4_d1)

        x1_d2 = self.inc_d2(x)
        x2_d2 = self.down1_d2(x1_d2)
        x3_d2 = self.down2_d2(x2_d2)
        x4_d2 = self.down3_d2(x3_d2)
        x5_d2 = self.down4_d2(x4_d2)

        x = self.up1(x5_d2, x4_d2)
        x = self.up2(x, x3_d2)
        x = self.up3(x, x2_d1)
        x = self.up4(x, x1_d1)

        # x = self.out_up(x)
        logits = self.outc(x)
        #################################
        return logits