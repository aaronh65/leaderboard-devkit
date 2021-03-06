import os, cv2
import numpy as np
import torch
import torch.nn.functional as F

from torchvision.models.segmentation import deeplabv3_resnet50

HAS_DISPLAY = int(os.environ.get('HAS_DISPLAY', 0))
SHOW_HEATMAPS = False
# n,c,h,w heatmaps
#@torch.no_grad()
#def logits2weights(logits, temperature):
#    logits = logits.clone().detach().cpu()/temperature # n,c,h,w
#    flat = logits.view(logits.shape[:-2] + (-1, ))
#    weights = F.softmax(flat, dim=-1).view_as(logits).numpy()
#    weights_max = np.amax(weights, axis=(-1,-2), keepdims=True)
#    weights_min = np.amin(weights, axis=(-1,-2), keepdims=True)
#    weights = (weights - weights_min) / (weights_max - weights_min)
#    return weights

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

class SegmentationModel(torch.nn.Module):
    def __init__(self, input_channels=3, n_steps=4, batch_norm=True, hack=False):
        super().__init__()

        self.hack = hack

        self.norm = torch.nn.BatchNorm2d(input_channels) if batch_norm else lambda x: x
        self.network = deeplabv3_resnet50(pretrained=False, num_classes=n_steps)
        self.extract = SpatialSoftmax()

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

        # extract 
        points, weights = self.extract(logits, temperature)

        return points, logits
