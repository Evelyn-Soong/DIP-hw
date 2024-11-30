import torch
import torch.nn as nn
import torch.nn.functional as F


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()
        self.upconv_relu = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.LeakyReLU(inplace=True)
        )

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x, is_drop=False):
        x = self.upconv_relu(x)
        x = self.bn(x)
        if is_drop:
            x = F.dropout2d(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsample, self).__init__()
        self.conv_relu = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3,
                      stride=2,
                      padding=1),
            nn.LeakyReLU(inplace=True)
        )

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x, is_bn=True):
        x = self.conv_relu(x)
        if is_bn:
            x = self.bn(x)
        return x


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # Encoder (Convolutional Layers)
        self.down1 = Downsample(3, 64)  # 3, 256, 256 ——> 64, 128, 128
        self.down2 = Downsample(64, 64 * 2)  # 64, 128, 128 ——> 128, 64, 64
        self.down3 = Downsample(64 * 2, 64 * 4)  # 128, 64, 64 ——> 256, 32, 32
        self.down4 = Downsample(64 * 4, 64 * 8)  # 256, 32, 32 ——> 512, 16, 16
        self.down5 = Downsample(64 * 8, 64 * 8)  # 512, 16, 16 ——> 512, 8, 8
        self.down6 = Downsample(64 * 8, 64 * 8)  # 512, 8, 8 ——> 512, 4, 4
        self.down7 = Downsample(64 * 8, 64 * 8)  # 512, 4, 4 ——> 512, 2, 2
        self.down8 = Downsample(64 * 8, 64 * 8)  # 512, 2, 2 ——> 512, 1, 1

        # Decoder (Deconvolutional Layers)
        self.up1 = Upsample(64 * 8, 64 * 8)  # 512, 2, 2
        self.up2 = Upsample(64 * 8 * 2, 64 * 8)  # 512, 4, 4
        self.up3 = Upsample(64 * 8 * 2, 64 * 8)  # 512, 8, 8
        self.up4 = Upsample(64 * 8 * 2, 64 * 8)  # 512, 16, 16
        self.up5 = Upsample(64 * 8 * 2, 64 * 4)  # 256, 32, 32
        self.up6 = Upsample(64 * 4 * 2, 64 * 2)  # 128, 64, 64
        self.up7 = Upsample(64 * 2 * 2, 64)  # 64, 128, 128
        self.up8 = nn.ConvTranspose2d(64 * 2, 3, kernel_size=3, stride=2, padding=1, output_padding=1)  # 3, 256, 256

    def forward(self, x):
        # Encoder forward pass
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        # Decoder forward pass
        u1 = self.up1(d8)
        u1 = torch.cat([u1, d7], dim=1)
        u2 = self.up2(u1)
        u2 = torch.cat([u2, d6], dim=1)
        u3 = self.up3(u2)
        u3 = torch.cat([u3, d5], dim=1)
        u4 = self.up4(u3)
        u4 = torch.cat([u4, d4], dim=1)
        u5 = self.up5(u4)
        u5 = torch.cat([u5, d3], dim=1)
        u6 = self.up6(u5)
        u6 = torch.cat([u6, d2], dim=1)
        u7 = self.up7(u6)
        u7 = torch.cat([u7, d1], dim=1)
        u8 = self.up8(F.relu(u7))

        return torch.tanh(u8)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.down1 = Downsample(6, 64)  # 64*128*128
        self.down2 = Downsample(64, 128)  # 128*64*64
        self.conv1 = nn.Conv2d(128, 256, 3)  # Conv2d——> 256*64*64
        self.bn = nn.BatchNorm2d(256)
        self.last = nn.Conv2d(256, 1, 3)

    def forward(self, anno, img):
        x = torch.cat([anno, img], axis=1)
        x = self.down1(x, is_bn=False)

        x = self.down2(x)
        x = F.leaky_relu(self.conv1(x))
        x = F.dropout2d(self.bn(x))

        x = torch.sigmoid(self.last(x))

        return x
