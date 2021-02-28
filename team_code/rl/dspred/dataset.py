from pathlib import Path

import numpy as np
import torch
import imgaug.augmenters as iaa
import pandas as pd

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from numpy import nan

from lbc.carla_project.src.converter import Converter, PIXELS_PER_WORLD
from lbc.carla_project.src.dataset_wrapper import Wrap
from lbc.carla_project.src.common import *


# Reproducibility.
#np.random.seed(0)
#torch.manual_seed(0)

# Data has frame skip of 5.
GAP = 1
STEPS = 4
N_CLASSES = len(COLOR)


def get_weights(data, key='speed', bins=4):
    if key == 'none':
        return [1 for _ in range(sum(len(x) for x in data))]
    elif key == 'even':
        values = np.hstack([[i for _ in range(len(x))] for i, x in enumerate(data)])
        bins = len(data)
    else:
        values = np.hstack(tuple(x.measurements[key].values[:len(x)] for x in data))
        values[np.isnan(values)] = np.mean(values[~np.isnan(values)])

    counts, edges = np.histogram(values, bins=bins)
    class_weights = counts.sum() / (counts + 1e-6)
    classes = np.digitize(values, edges[1:-1])

    print(counts)

    return class_weights[classes]


def get_dataset(dataset_dir, is_train=False, batch_size=4, num_workers=4, sample_by='none', **kwargs):
    data = list()
    transform = transforms.Compose([
        get_augmenter() if is_train else lambda x: x,
        transforms.ToTensor()
        ])

    episodes = list(sorted(Path(dataset_dir).glob('*')))

    for i, _dataset_dir in enumerate(episodes):
        add = False
        add |= (is_train and i % 10 < 9)
        add |= (not is_train and i % 10 >= 9)

        if add:
            data.append(CarlaDataset(_dataset_dir, transform, **kwargs))

    print('%d frames.' % sum(map(len, data)))

    weights = torch.DoubleTensor(get_weights(data, key=sample_by))
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
    data = torch.utils.data.ConcatDataset(data)

    return Wrap(data, sampler, batch_size, 1000 if is_train else 100, num_workers)


def get_augmenter():
    seq = iaa.Sequential([
        iaa.Sometimes(0.05, iaa.GaussianBlur((0.0, 1.3))),
        iaa.Sometimes(0.05, iaa.AdditiveGaussianNoise(scale=(0.0, 0.05 * 255))),
        iaa.Sometimes(0.05, iaa.Dropout((0.0, 0.1))),
        iaa.Sometimes(0.10, iaa.Add((-0.05 * 255, 0.05 * 255), True)),
        iaa.Sometimes(0.20, iaa.Add((0.25, 2.5), True)),
        iaa.Sometimes(0.05, iaa.contrast.LinearContrast((0.5, 1.5))),
        iaa.Sometimes(0.05, iaa.MultiplySaturation((0.0, 1.0))),
        ])

    return seq.augment_image


# https://github.com/guopei/PoseEstimation-FCN-Pytorch/blob/master/heatmap.py
def make_heatmap(size, pt, sigma=8):
    img = np.zeros(size, dtype=np.float32)
    pt = [
            np.clip(pt[0], sigma // 2, img.shape[1]-sigma // 2),
            np.clip(pt[1], sigma // 2, img.shape[0]-sigma // 2)
            ]

    # Check that any part of the gaussian is in-bounds
    ul = [int(pt[0] - 3 * sigma), int(pt[1] - 3 * sigma)]
    br = [int(pt[0] + 3 * sigma + 1), int(pt[1] + 3 * sigma + 1)]

    # Generate gaussian
    size = 6 * sigma + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2

    # The gaussian is not normalized, we want the center value to equal 1
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]

    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

    return img


def preprocess_semantic(semantic_np):
    topdown = CONVERTER[semantic_np]
    topdown = torch.LongTensor(topdown)
    topdown = torch.nn.functional.one_hot(topdown, N_CLASSES).permute(2, 0, 1).float()

    return topdown


class CarlaDataset(Dataset):
    def __init__(self, dataset_dir, transform=transforms.ToTensor(), n=0):
        dataset_dir = Path(dataset_dir)
        measurements = list(sorted((dataset_dir / 'measurements').glob('*.json')))

        self.transform = transform
        self.dataset_dir = dataset_dir
        self.frames = list()
        self.measurements = pd.DataFrame([eval(x.read_text()) for x in measurements[:120]])
        self.converter = Converter()

        # n-step returns
        self.n = n

        print(dataset_dir)
        print(self.measurements)

        for image_path in sorted((dataset_dir / 'topdown').glob('*.png'))[:120]:
            frame = str(image_path.stem)

            #assert (dataset_dir / 'rgb_left' / ('%s.png' % frame)).exists()
            #assert (dataset_dir / 'rgb_right' / ('%s.png' % frame)).exists()
            assert (dataset_dir / 'topdown' / ('%s.png' % frame)).exists()
            assert int(frame) < len(self.measurements)

            self.frames.append(frame)

        assert len(self.frames) > 0, '%s has 0 frames.' % dataset_dir

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, i):
        path = self.dataset_dir

        frame = self.frames[i]
        topdown = Image.open(path / 'topdown' / ('%s.png' % frame))
        topdown = topdown.crop((128, 0, 128 + 256, 256))
        topdown = preprocess_semantic(np.array(topdown))
        tick_data = self.measurements.iloc[i]
        target = torch.FloatTensor(np.float32((tick_data['x_tgt'], tick_data['y_tgt'])))
        action = torch.FloatTensor(np.float32((tick_data['x_aim'], tick_data['y_aim'])))

        ni = i + self.n + 1
        ni = min(ni, len(self.frames))
        reward = self.measurements.loc[i:ni, 'reward'].sum()
        reward = torch.FloatTensor([reward])
        ntick_data = self.measurements.iloc[ni-1]
        done = torch.FloatTensor([ntick_data['done']])

        nframe = self.frames[min(ni, len(self.frames)-1)] # 'next_state' at done edge case
        ntopdown = Image.open(path / 'topdown' / ('%s.png' % nframe))
        ntopdown = ntopdown.crop((128, 0, 128 + 256, 256))
        ntopdown = preprocess_semantic(np.array(ntopdown))
        ntarget = torch.FloatTensor(np.float32((ntick_data['x_tgt'], ntick_data['y_tgt'])))

        
        #print(f'{frame} {nframe}')
        return (topdown, target), action, reward, (ntopdown, ntarget), done
        
if __name__ == '__main__':
    import sys
    import cv2
    import argparse
    from PIL import ImageDraw
    from lbc.carla_project.src.utils.heatmap import ToHeatmap

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--n', type=int, default=0)
    args = parser.parse_args()

    data = CarlaDataset(args.path, n=args.n)

    converter = Converter()
    to_heatmap = ToHeatmap()

    for i in range(len(data)):
        state, action, reward, next_state, done = data[i]

        # this state
        topdown, target = state
        _topdown = COLOR[topdown.argmax(0).cpu().numpy()]
        heatmap = to_heatmap(target[None], topdown[None]).squeeze()
        _heatmap = heatmap.cpu().squeeze().numpy() / 10.0 + 0.9
        _topdown[heatmap > 0.1] = 255
        _topdown = Image.fromarray(_topdown)
        _draw = ImageDraw.Draw(_topdown)
        x, y = action.cpu().squeeze().numpy().astype(np.uint8)
        _draw.ellipse((x-2, y-2, x+2, y+2), (0,255,0))
        _draw.text((5, 10), f'reward = {reward.item():.5f}', (255,255,255))
        _topdown = cv2.cvtColor(np.array(_topdown), cv2.COLOR_BGR2RGB)

        # next state
        ntopdown, ntarget = next_state
        _ntopdown = COLOR[ntopdown.argmax(0).cpu().numpy()]
        nheatmap = to_heatmap(ntarget[None], ntopdown[None]).squeeze()
        _nheatmap = nheatmap.cpu().squeeze().numpy() / 10.0 + 0.9
        _ntopdown[nheatmap > 0.1] = 255
        _ntopdown = cv2.cvtColor(_ntopdown, cv2.COLOR_BGR2RGB)

        _combined = np.hstack((_topdown, _ntopdown))
        cv2.imshow('topdown', _combined)
        cv2.waitKey(50)

    #for i in range(len(data)):
    #    state, action, reward, next_state, done = batch

    #for i in range(len(data)):
    #    rgb, topdown, points, target, actions, meta = data[i]
    #    points_unnormalized = (points + 1) / 2 * 256
    #    points_cam = converter(points_unnormalized)

    #    target_cam = converter(target)

    #    heatmap = to_heatmap(target[None], topdown[None]).squeeze()
    #    heatmap_cam = to_heatmap(target_cam[None], rgb[None]).squeeze()

    #    _heatmap = heatmap.cpu().squeeze().numpy() / 10.0 + 0.9
    #    _heatmap_cam = heatmap_cam.cpu().squeeze().numpy() / 10.0 + 0.9

    #    _rgb = (rgb.cpu() * 255).byte().numpy().transpose(1, 2, 0)[:, :, :3]
    #    _rgb[heatmap_cam > 0.1] = 255
    #    _rgb = Image.fromarray(_rgb)

    #    _topdown = COLOR[topdown.argmax(0).cpu().numpy()]
    #    _topdown[heatmap > 0.1] = 255
    #    _topdown = Image.fromarray(_topdown)
    #    _draw_map = ImageDraw.Draw(_topdown)
    #    _draw_rgb = ImageDraw.Draw(_rgb)

    #    for x, y in points_unnormalized:
    #        _draw_map.ellipse((x-2, y-2, x+2, y+2), (255, 0, 0))

    #    for x, y in converter.cam_to_map(points_cam):
    #        _draw_map.ellipse((x-1, y-1, x+1, y+1), (0, 255, 0))

    #    for x, y in points_cam:
    #        _draw_rgb.ellipse((x-2, y-2, x+2, y+2), (255, 0, 0))

    #    _topdown.thumbnail(_rgb.size)

    #    cv2.imshow('debug', cv2.cvtColor(np.hstack((_rgb, _topdown)), cv2.COLOR_BGR2RGB))
    #    cv2.waitKey(0)
