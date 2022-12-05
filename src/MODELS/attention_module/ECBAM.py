import torch
import torch.nn as nn
from torch.nn import functional as F


## help functions ##
class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x


class ChannelPool(nn.Module):
    def forward(self, x):
        max_pool = torch.max(x, dim=1)[0].unsqueeze(1)
        avg_pool = torch.mean(x, dim=1).unsqueeze(1)
        concat_pool = torch.cat( (max_pool, avg_pool), dim=1 )

        return concat_pool


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
                Flatten(),
                nn.Linear(gate_channels, gate_channels // reduction_ratio),
                nn.ReLU(),
                nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None

        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool3d(x, (x.size(2), x.size(3), x.size(4)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool3d(x, (x.size(2), x.size(3), x.size(4)))
                channel_att_raw = self.mlp( max_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum += channel_att_raw

            channel_att_sum += channel_att_raw

        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).unsqueeze(4).expand_as(x)
        return x * scale


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = nn.Sequential(
            nn.Conv2d(
                in_channels=2,
                out_channels=1,
                kernel_size=kernel_size,
                stride=1,
                padding=(kernel_size-1) // 2,
                bias=False,
            ),
           nn.BatchNorm2d(1, eps=1e-5, momentum=0.01, affine=True)
        )

    def forward(self, x): ## x.shape = [2, 32, 100, 120, 140]
        compress = torch.mean(x, dim=1) ## compress.shape = [2, 100, 120, 140]
        compress1_mean = torch.mean(compress, dim=1).unsqueeze(1) ## compress1.shape = [2, 1, 120, 140]
        compress2_mean = torch.mean(compress, dim=2).unsqueeze(1) ## compress2.shape = [2, 1, 100, 140]
        # compress3_mean = torch.mean(compress, dim=3).unsqueeze(1) ## compress3.shape = [2, 1, 100, 120]

        compress1_max = torch.max(compress, dim=1)[0].unsqueeze(1) ## compress1.shape = [2, 1, 120, 140]
        compress2_max = torch.max(compress, dim=2)[0].unsqueeze(1) ## compress2.shape = [2, 1, 100, 140]
        # compress3_max = torch.max(compress, dim=3)[0].unsqueeze(1) ## compress3.shape = [2, 1, 100, 120]

        compress1 = self.spatial(torch.cat( (compress1_max, compress1_mean), dim=1 ))
        compress2 = self.spatial(torch.cat( (compress2_max, compress2_mean), dim=1 ))
        # compress3 = self.spatial(torch.cat( (compress3_max, compress3_mean), dim=1 ))

        scale1 = torch.sigmoid(compress1).unsqueeze(2).expand_as(x)
        scale2 = torch.sigmoid(compress2).unsqueeze(3).expand_as(x)
        # scale3 = torch.sigmoid(compress3).unsqueeze(4).expand_as(x)

        return x * scale1 * scale2


class ECBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=8, pool_types=['avg', 'max'], spatial=True):
        super(ECBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)

        if spatial: 
            self.SpatialGate = SpatialGate()
        else: 
            self.SpatialGate = None

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if self.SpatialGate is not None:
            x_out = self.SpatialGate(x_out)
        return x_out


if __name__ == "__main__":
    att = SpatialGate()
    att.cuda()
    
    x = torch.randn(size=(2, 32, 100, 120, 140), device='cuda:0')
    # x = torch.randn(size=(2, 32, 64, 64), device='cuda:0')

    print(att(x).shape)

    # import torchsummary
    # model.cuda()
    # torchsummary.summary(model, (2, 164, 164, 212))