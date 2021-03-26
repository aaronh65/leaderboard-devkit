import os, cv2
import numpy as np
import torch
import torch.nn.functional as F

from torchvision.models.segmentation import deeplabv3_resnet50

HAS_DISPLAY = int(os.environ.get('HAS_DISPLAY', 0))
SHOW_HEATMAPS = False
# n,c,h,w heatmaps
@torch.no_grad()
def make_heatmap_probs(heatmaps, temperature):
    heatmaps = heatmaps.clone().detach().cpu()/temperature # 4, H, W
    flat = heatmaps.view(heatmaps.shape[:-2] + (-1, ))
    softmax = F.softmax(flat, dim=-1)
    softmax = softmax.view_as(heatmaps).numpy()
    softmax_max = np.amax(softmax, axis=(-1,-2), keepdims=True)
    softmax_min = np.amin(softmax, axis=(-1,-2), keepdims=True)
    softmax = (softmax - softmax_min) / (softmax_max - softmax_min)
    return softmax

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
    def forward(self, logit, temperature):
        """
        Assumes logits is size (n, c, h, w)
        """
        flat = logit.view(logit.shape[:-2] + (-1,))
        weights = F.softmax(flat / temperature, dim=-1).view_as(logit)

        x = (weights.sum(-2) * torch.linspace(-1, 1, logit.shape[-1]).type_as(logit)).sum(-1)
        y = (weights.sum(-1) * torch.linspace(-1, 1, logit.shape[-2]).type_as(logit)).sum(-1)

        return torch.stack((x, y), -1)

# argmax isn't usually differentiable but we can hack together 
# an implementation that is
class SpatialArgmax(torch.nn.Module):
    def forward(self, logit, temperature):
        """
        Assumes logits is size (n, c, h, w)
        """
        #n,c,h,w = logit.shape
        #flat = logit.view((n,c,h*w)) # (N,C,H*W)
        #Q_all, _ = torch.max(flat, -1, keepdim=True) # (N,C,1)

        # create mask
        #flat = flat / Q_all
        #flat[flat < 1] = flat[flat < 1] * 1e-4
        #flat[flat < Q_all] = flat[flat < Q_all] / Q_all / 1e-4
        #flat[flat == Q_all] = flat[flat == Q_all] / Q_all 
        #masked = flat.view((n,c,h,w))
        out = SpatialSoftmax()(logit, temperature)

        return out

class SegmentationModel(torch.nn.Module):
    def __init__(self, input_channels=3, n_steps=4, mode='expectation', batch_norm=True, hack=False):
        super().__init__()

        assert mode in ['expectation', 'argmax'], 'invalid mode'

        self.hack = hack

        self.norm = torch.nn.BatchNorm2d(input_channels) if batch_norm else lambda x: x
        self.network = deeplabv3_resnet50(pretrained=False, num_classes=n_steps)
        self.extract = SpatialSoftmax() if mode is 'expectation' else SpatialArgmax()

        old = self.network.backbone.conv1
        self.network.backbone.conv1 = torch.nn.Conv2d(
                input_channels, old.out_channels,
                kernel_size=old.kernel_size, stride=old.stride,
                padding=old.padding, bias=old.bias)

    def forward(self, input, heatmap=False, temperature=10):
        if self.hack:
            input = torch.nn.functional.interpolate(input, scale_factor=0.5, mode='bilinear')

        x = self.norm(input)
        x = self.network(x)['out']

        if self.hack:
            x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode='bilinear')

        if HAS_DISPLAY and SHOW_HEATMAPS:
            heatmap_probs = make_heatmap_probs(x, temperature)
            hmap_show = np.concatenate([hmap for hmap in heatmap_probs[0]], axis=1)
            cv2.imshow('hmap', hmap_show)
            cv2.waitKey(1)

        y = self.extract(x, temperature)

        if heatmap:
            return y, x

        return y
