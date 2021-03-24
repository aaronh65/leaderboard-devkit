import cv2
import torch
import torch.nn.functional as F

from torchvision.models.segmentation import deeplabv3_resnet50


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

class SpatialArgmax(torch.nn.Module):
    def forward(self, logit, temperature):
        """
        Assumes logits is size (n, c, h, w)
        """
        n,c,h,w = logit.shape
        flat = logit.view((n,c,h*w)) # (N,C,H*W)
        Q_all, action_flat = torch.max(flat, -1, keepdim=True) # (N,C,1)
        # x,y coordinates
        action = torch.cat((action_flat % w, action_flat // w), axis=2).type_as(logit) # (N,C,2)
        action = (action / 255 - 1/2) * 2
        #return action, Q_all # (N,4,2), (N,4,1)
        return action # (N,4,2)

class SegmentationModel(torch.nn.Module):
    def __init__(self, input_channels=3, n_steps=4, mode='expectation', batch_norm=True, hack=False, temperature=1.0):
        super().__init__()

        assert mode in ['expectation', 'argmax'], 'invalid mode'

        self.temperature = temperature
        self.hack = hack

        self.norm = torch.nn.BatchNorm2d(input_channels) if batch_norm else lambda x: x
        self.network = deeplabv3_resnet50(pretrained=False, num_classes=n_steps)
        self.extract = SpatialSoftmax() if mode is 'expectation' else SpatialArgmax()

        old = self.network.backbone.conv1
        self.network.backbone.conv1 = torch.nn.Conv2d(
                input_channels, old.out_channels,
                kernel_size=old.kernel_size, stride=old.stride,
                padding=old.padding, bias=old.bias)

    def forward(self, input, heatmap=False):
        if self.hack:
            input = torch.nn.functional.interpolate(input, scale_factor=0.5, mode='bilinear')

        x = self.norm(input)
        x = self.network(x)['out']

        if self.hack:
            x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode='bilinear')
        y = self.extract(x, self.temperature)

        if heatmap:
            return y, x

        return y
