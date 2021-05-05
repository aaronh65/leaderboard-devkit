import os, cv2
import numpy as np
import torch
import torch.nn.functional as F

from torchvision.models.segmentation import deeplabv3_resnet50
from dqn.src.agents.deeplabv3 import ValueHead, ASPPExtract
#from torchvision.models.segmentation.deeplabv3 import ASPP

HAS_DISPLAY = int(os.environ.get('HAS_DISPLAY', 0))
SHOW_HEATMAPS = False

@torch.no_grad()
# assumes n,c,h,w
def spatial_norm(tensor):
    n,c,h,w = tensor.shape
    flat = tensor.view((n,c,h*w))
    norm_max, _ = torch.max(flat, dim=-1, keepdim=True)
    norm_min, _ = torch.min(flat, dim=-1, keepdim=True)
    flat = (flat - norm_min) / (norm_max - norm_min)
    out = flat.view_as(tensor)
    return out # n,c,h,w

'''
    outputs values for discrete actions
    - default throttle: [0.0, 0.1,..., 0.9. 1.0]
    - default steer:    [-1.0 -.9,...,-.1,0,.1,...,.9,1.0]
    11 throttle choices * 21 steer choices = 231 actions
'''
class DiscreteController(torch.nn.Module):
    def __init__(self, n_input=4, n_steer=21, n_speed=11, k=128):
        super().__init__()

        self.layers = torch.nn.Sequential(
                torch.nn.BatchNorm1d(n_input * 2),
                torch.nn.Linear(n_input * 2, k), 
                torch.nn.ReLU(),

                torch.nn.BatchNorm1d(k),
                torch.nn.Linear(k, k), 
                torch.nn.ReLU(),

                torch.nn.BatchNorm1d(k),
                torch.nn.Linear(k, n_steer*n_speed))

    def forward(self, x):
        return self.layers(torch.flatten(x, 1))

class RawController(torch.nn.Module):
    def __init__(self, n_input=4, k=32):
        super().__init__()

        self.layers = torch.nn.Sequential(
                torch.nn.BatchNorm1d(n_input * 2),
                torch.nn.Linear(n_input * 2, k), torch.nn.ReLU(),

                torch.nn.BatchNorm1d(k),
                torch.nn.Linear(k, k), torch.nn.ReLU(),

                torch.nn.BatchNorm1d(k),
                torch.nn.Linear(k, 2))

    def forward(self, x):
        return self.layers(torch.flatten(x, 1))


class SpatialSoftmax(torch.nn.Module):
    def forward(self, logits, temperature):
        """
        Assumes logits is size (n, c, h, w)
        """
        flat = logits.view(logits.shape[:-2] + (-1,))
        weights = F.softmax(flat / temperature, dim=-1).view_as(logits)

        x = (weights.sum(-2) * torch.linspace(-1, 1, logits.shape[-1]).type_as(logits)).sum(-1)
        y = (weights.sum(-1) * torch.linspace(-1, 1, logits.shape[-2]).type_as(logits)).sum(-1)

        return torch.stack((x, y), -1), weights

class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, bias=False):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)
        self.norm = torch.nn.BatchNorm2d(out_channels)

    def forward(self, x, relu=True):
        x = self.conv(x)
        x = self.norm(x)
        if relu:
            x = F.relu(x)
        return x

class SegmentationModel(torch.nn.Module):
    def __init__(self, input_channels=3, n_steps=4, batch_norm=True, hack=False, extract=False):
        super().__init__()

        self.hack = hack
        self.extract = extract
        if self.extract:
            self.extract_module = ASPPExtract(in_channels=64, latent_dim=16, num_classes=n_steps)
        out_channels = 64 if self.extract else n_steps

        self.norm = torch.nn.BatchNorm2d(input_channels) if batch_norm else lambda x: x
        self.network = deeplabv3_resnet50(pretrained=False, num_classes=out_channels)


        self.spatial_softmax = SpatialSoftmax()

        old = self.network.backbone.conv1
        self.network.backbone.conv1 = torch.nn.Conv2d(
                input_channels, old.out_channels,
                kernel_size=old.kernel_size, stride=old.stride,
                padding=old.padding, bias=old.bias)

    def forward(self, input, temperature=10):
        # downsampling input for smaller network?
        if self.hack:
            input = torch.nn.functional.interpolate(input, scale_factor=0.5, mode='bilinear')
        x = self.norm(input)
        logits = self.network(x)['out']
        if self.hack:
            logits = torch.nn.functional.interpolate(logits, scale_factor=2.0, mode='bilinear')

        # conv blocks to smooth out backbone upsampling artifacts
        if self.extract:
            logits = self.extract_module(logits)

        # extract 
        points, weights = self.spatial_softmax(logits, temperature)

        return points, logits, weights
