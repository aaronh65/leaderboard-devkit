import torch
from torch import nn
from torch.nn import functional as F


class ValueHead(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(ValueHead, self).__init__(
            #ASPP(in_channels, [1,3,5], out_channels=16),
            nn.Conv2d(64, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, num_classes, 3, padding=1),
        )


class ASPPExtract(nn.Module):
    def __init__(self, in_channels, latent_dim, num_classes, rates=list(range(1,10,1))):
        super().__init__()
        self.convs = []
        for i, rate in enumerate(rates):
            self.convs.append(ASPPConv(in_channels, latent_dim, rate))
        self.convs = nn.ModuleList(self.convs)
        self.smooth = ASPPConv(len(self.convs)*latent_dim, latent_dim, dilation=1)
        self.project = nn.Conv2d(latent_dim, num_classes, 1, bias=False)


    def forward(self, x):
        features = []
        for conv in self.convs:
            features.append(conv(x))
        features = torch.cat(features, dim=1)
        features = self.smooth(features)
        out = self.project(features)
        return out

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=256):
        super(ASPP, self).__init__()
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)
