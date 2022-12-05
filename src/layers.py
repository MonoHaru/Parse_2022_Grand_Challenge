import torch
import torch.nn as nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding, 
            bias=False,
            )
        self.norm = nn.InstanceNorm3d(out_channels, affine=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = F.relu(x, inplace=True)
        
        return x


class ConvTransBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.convTransposed = nn.ConvTranspose3d(
            in_channels=in_channels,
            out_channels=out_channels,
            bias=False,
            **kwargs,
        )

        # self.norm = nn.InstanceNorm3d(out_channels, affine=True)

    def forward(self, x):
        x = self.convTransposed(x)
        # x = self.norm(x)
        # x = F.relu(x, inplace=True)

        return x    


class SEBlock(nn.Module):
    class SEWeights(nn.Module):
        def __init__(self, in_channels, reduction=4):
            super().__init__()
            self.conv1 = nn.Conv3d(in_channels, in_channels // reduction, kernel_size=1, stride=1, padding=0, bias=True)
            self.conv2 = nn.Conv3d(in_channels // reduction, in_channels, kernel_size=1, stride=1, padding=0, bias=True)

        def forward(self, x):
            b, c, d, h, w = x.size()
            out = torch.mean(x.view(b, c, -1), dim=-1).view(b, c, 1, 1, 1)  # output_shape: in_channels x (1, 1, 1)
            out = F.relu(self.conv1(out))
            out = self.conv2(out)
            return out

    def __init__(self, in_channels, reduction=2):
        super().__init__()
        self.norm = nn.InstanceNorm3d(in_channels, affine=False)
        self.gamma = self.SEWeights(in_channels, reduction)
        self.beta = self.SEWeights(in_channels, reduction)

    def forward(self, x):
        gamma = torch.sigmoid(self.gamma(x))
        beta = torch.tanh(self.beta(x))
        x = self.norm(x)
        return gamma * x + beta


class SEConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, reduction=32):
        super().__init__()

        self.conv1 = nn.Conv3d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=3, 
            stride=stride, 
            padding=1, 
            bias=False,
            )
        self.norm1 = nn.InstanceNorm3d(out_channels, affine=True)
        self.conv2 = nn.Conv3d(
            in_channels=out_channels, 
            out_channels=out_channels, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            bias=False,
        )
        self.norm2 = nn.InstanceNorm3d(out_channels, affine=True)
        self.senorm = SEBlock(in_channels=out_channels, reduction=reduction)
        
        if in_channels != out_channels:
            self.projection = nn.Sequential(
                ConvBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=0),
                nn.InstanceNorm3d(out_channels, affine=True)
            )
        else:
            self.projection = nn.Sequential()

    def forward(self, x):
        x = F.relu(self.norm1(self.conv1(x)), inplace=True)        
        x = self.norm2(self.conv2(x))
        x = self.senorm(x)
        x = F.relu(x, inplace=True)

        return x


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels,  scale=2, reduction=2):
        super().__init__()
        self.scale = scale
        self.conv = nn.Conv3d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=1, 
            stride=1, 
            padding=0, 
            bias=False,
            )
        self.norm = nn.InstanceNorm3d(out_channels, affine=True)
        self.senorm = SEBlock(in_channels=out_channels, reduction=reduction)
        self.upsample = nn.Upsample(scale_factor=scale, mode='trilinear', align_corners=True)

    def forward(self, x):
        x = F.relu(self.norm(self.conv(x)), inplace=True)
        x = self.upsample(x)
        return x