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
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        
        return x


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2)
        self.conv = nn.Conv3d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding, 
            bias=False,
            )
        self.norm = nn.InstanceNorm3d(out_channels, affine=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)

        return x  


class UNet(nn.Module):
    def __init__(self, in_channels, n_cls, n_filters):
        super().__init__()

        self.n_cls = n_cls

        self.left1 = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=n_filters),
            ConvBlock(in_channels=n_filters, out_channels=n_filters),
        )

        self.left2 = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2),
            ConvBlock(in_channels=n_filters, out_channels=n_filters * 2),
            ConvBlock(in_channels=n_filters * 2, out_channels=n_filters * 2),
        )

        self.left3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2),
            ConvBlock(in_channels=n_filters * 2, out_channels=n_filters * 4),
            ConvBlock(in_channels=n_filters * 4, out_channels=n_filters * 4),
        )

        self.center = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2),
            ConvBlock(in_channels=n_filters * 4, out_channels=n_filters * 8),
            ConvBlock(in_channels=n_filters * 8, out_channels=n_filters * 8),
        )

        self.upsample3 = UpsampleBlock(in_channels=n_filters * 8, out_channels=n_filters * 4)
        self.right3 = nn.Sequential(
            ConvBlock(in_channels=n_filters * 8, out_channels=n_filters * 4),
            ConvBlock(in_channels=n_filters * 4, out_channels=n_filters * 4),
        )

        self.upsample2 = UpsampleBlock(in_channels=n_filters * 4, out_channels=n_filters * 2)
        self.right2 = nn.Sequential(
            ConvBlock(in_channels=n_filters * 4, out_channels=n_filters * 2),
            ConvBlock(in_channels=n_filters * 2, out_channels=n_filters * 2),
        )

        self.upsample1 = UpsampleBlock(in_channels=n_filters * 2, out_channels=n_filters * 1)
        self.right1 = nn.Sequential(
            ConvBlock(in_channels=n_filters * 2, out_channels=n_filters * 1),
            ConvBlock(in_channels=n_filters * 1, out_channels=n_filters * 1),
        )

        self.score = nn.Conv3d(n_filters, self.n_cls, kernel_size=1, stride=1, padding=0)

    def forward(self, x): 
        left1 = self.left1(x) 
        left2 = self.left2(left1)
        left3 = self.left3(left2)
        x = self.center(left3)

        x = self.right3(torch.cat([self.upsample3(x), left3], 1))
        x = self.right2(torch.cat([self.upsample2(x), left2], 1))
        x = self.right1(torch.cat([self.upsample1(x), left1], 1))
        x = self.score(x)

        if self.n_cls == 1:
            return torch.sigmoid(x)
        else:
            return F.softmax(x, dim=1)

if __name__ == "__main__":
    size=(1, 1, 320, 320, 240)
    x = torch.randn(size=size, device='cpu')
    model = UNet(in_channels=1, n_cls=1, n_filters=16)
    model.to('cpu')

    print(model(x).shape)