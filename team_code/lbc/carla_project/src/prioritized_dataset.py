from pathlib import Path

import cv2
import numpy as np
import torch
import imgaug.augmenters as iaa
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from numpy import nan

from lbc.carla_project.src.converter import Converter, PIXELS_PER_WORLD
from lbc.carla_project.src.dataset_wrapper import Wrap
from lbc.carla_project.src import common


# Reproducibility.
#np.random.seed(0)
#torch.manual_seed(0)

# Data has frame skip of 5.
GAP = 1
STEPS = 4
N_CLASSES = len(common.COLOR)


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


def get_dataset(hparams, is_train=True, batch_size=128, num_workers=4, sample_by='none', **kwargs):
    dataset = CarlaDataset(hparams, is_train)
    loader = DataLoader(dataset, batch_size=hparams.batch_size, num_workers=num_workers, pin_memory=True, drop_last=True)
    return loader


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
    topdown = common.CONVERTER[semantic_np]
    topdown = torch.LongTensor(topdown)
    topdown = torch.nn.functional.one_hot(topdown, N_CLASSES).permute(2, 0, 1).float()

    return topdown


class CarlaDataset(Dataset):
    def __init__(self, hparams, is_train):

        dataset_dir = hparams.dataset_dir / 'data'
        routes = dataset_dir.glob('*')
        routes = sorted([path for path in routes if path.is_dir() and 'logs' not in str(path)])

        episodes = list()
        for route in routes:
            episodes.extend(sorted(route.glob('*')))

        self.frames = list()
        measurements = list()
        for i, _dataset_dir in enumerate(episodes): # 90-10 train/val
            add = False
            add |= (is_train and i % 10 < 9)
            add |= (not is_train and i % 10 >= 9)

            # TODO: need to account for the last four frames of each dataset run
            if add:
                paths = sorted((_dataset_dir / 'rgb').glob('*.png'))
                end = len(paths) - GAP*STEPS
                for image_path in paths[:end]:
                    frame = str(image_path.stem)
                    assert (_dataset_dir / 'topdown' / ('%s.png' % frame)).exists()
                    self.frames.append(_dataset_dir / frame)

                measurements.extend(list(sorted((_dataset_dir / 'measurements').glob('*.json'))))

        self.dataset_dir = dataset_dir
        self.measurements = pd.DataFrame([eval(x.read_text()) for x in measurements])
        self.converter = Converter()

        self.hparams = hparams

        self.num_frames = len(self.frames)
        self.len = hparams.steps_per_epoch if is_train else hparams.steps_per_epoch // 10

        # loss weights for prioritized sampling
        self.weights = [1e-9] * self.num_frames
        
        assert len(self.frames) > 0, '%s has 0 frames.' % dataset_dir

    def __len__(self):
        #return self.num_frames
        return self.num_frames

    def _recompute_weights(self):
        pass

    def __getitem__(self, i):

        #i = np.random.randint(self.num_frames)
        print(i)

        frame = self.frames[i].stem
        root = self.frames[i].parent
        route, rep = root.parts[-2:]

        meta = '%s %s %s' % (route, rep, frame)

        rgb = Image.open(root / 'rgb' / ('%s.png' % frame))
        rgb = transforms.functional.to_tensor(rgb)

        #rgb_left = Image.open(root / 'rgb_left' / ('%s.png' % frame))
        #rgb_left = transforms.functional.to_tensor(rgb_left)

        #rgb_right = Image.open(root / 'rgb_right' / ('%s.png' % frame))
        #rgb_right = transforms.functional.to_tensor(rgb_right)

        topdown = Image.open(root / 'topdown' / ('%s.png' % frame))
        #topdown = topdown.crop((128, 0, 128 + 256, 256))
        #topdown = np.array(topdown)
        #topdown = preprocess_semantic(topdown)

        u = np.float32(self.measurements.iloc[i][['x', 'y']])
        theta = self.measurements.iloc[i]['theta']
        if np.isnan(theta):
            theta = 0.0
        theta = theta + np.pi / 2
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)],
            ])

        points = list()

        for skip in range(1, STEPS+1):
            j = i + GAP * skip
            v = np.array(self.measurements.iloc[j][['x', 'y']])

            target = R.T.dot(v - u)
            target *= PIXELS_PER_WORLD
            target += [128, 256]

            points.append(target)

        points = torch.FloatTensor(points)
        points = torch.clamp(points, 0, 256)

        target = np.float32(self.measurements.iloc[i][['x_command', 'y_command']])
        target = R.T.dot(target - u)
        target *= PIXELS_PER_WORLD
        target += [128, 256]
        target = np.clip(target, 0, 256)

        topdown, points, target = self._augment_and_preprocess(topdown, points, target)
        points = (points / 256) * 2 - 1
        points = torch.FloatTensor(points)
        target = torch.FloatTensor(target)

        # heatmap = make_heatmap((256, 256), target)
        # heatmap = torch.FloatTensor(heatmap).unsqueeze(0)

        # command_img = self.converter.map_to_cam(torch.FloatTensor(target))
        # heatmap_img = make_heatmap((144, 256), command_img)
        # heatmap_img = torch.FloatTensor(heatmap_img).unsqueeze(0)

        actions = np.float32(self.measurements.iloc[i][['steer', 'target_speed']])
        actions[np.isnan(actions)] = 0.0
        actions = torch.FloatTensor(actions)
        info = {
                'meta':
                }
        meta = torch.FloatTensor(encode_str(meta))

        return rgb, topdown, points, target, actions, meta

    def _augment_and_preprocess(self, topdown, points, target):
        
        dr = np.random.randint(-self.hparams.angle_jitter, self.hparams.angle_jitter+1)
        dx = np.random.randint(-self.hparams.pixel_jitter, self.hparams.pixel_jitter+1)
        dy = np.random.randint(0, self.hparams.pixel_jitter+1)

        offset = np.ones(points.shape)
        offset.T[0] = dx
        offset.T[1] = dy
        pixel_rotation = cv2.getRotationMatrix2D((256//2,256), dr, 1) # 2x3

        points = np.hstack((points, np.ones((4,1)))) # 4x3
        points = np.matmul(pixel_rotation, points.T).T # 4x2
        points = np.clip(points - offset, 0, 255)

        target = np.array((target[0], target[1], 1)) # 1x3
        target = np.matmul(pixel_rotation, target.T).T # 1x2
        target = target.squeeze() # 2
        
        image_rotation = cv2.getRotationMatrix2D((256,256), dr, 1) # 2x3

        topdown = np.array(topdown)
        topdown = cv2.warpAffine(topdown, image_rotation, 
                topdown.shape[1::-1], flags=cv2.INTER_NEAREST)
        topdown = Image.fromarray(topdown)
        topdown = topdown.crop((128+dx,0+dy,128+256+dx,256+dy))
        topdown = preprocess_semantic(np.array(topdown))

        return topdown, points, target

class Wrap(object):
    def __init__(self, dataloader, num_samples):
        self.num_samples = num_samples
        self.loader = dataloader
        self.iter_loader = iter(self.loader)
        self.idx = 0

    def __iter__(self):
        return self

    def __len__(self):
        return self.num_samples

    def __next__(self):
        if self.idx == self.num_samples:
            self.idx = 0
            raise StopIteration
        else:
            self.idx += 1
        try:
            return next(self.iter_loader)
        except StopIteration:
            self.iter_loader = iter(self.loader)
            return next(self.iter_loader)


if __name__ == '__main__':
    import sys
    import argparse
    from PIL import ImageDraw
    from lbc.carla_project.src.utils.heatmap import ToHeatmap

    # for path in sorted(Path('/home/bradyzhou/data/carla/carla_challenge_curated').glob('*')):
        # data = CarlaDataset(path)

        # for i in range(len(data)):
            # data[i]

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=Path, required=True)
    parser.add_argument('--angle_jitter', type=float, default=5)
    parser.add_argument('--pixel_jitter', type=int, default=5.5) # 3 meters
    parser.add_argument('--steps_per_epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=4)
    hparams = parser.parse_args()

    loader = get_dataset(hparams, is_train=True)
    converter = Converter()
    to_heatmap = ToHeatmap()

    for i, data in enumerate(loader):
        rgb, topdown, points, target, actions, meta = [_data[0] for _data in data] # first elem
        points_unnormalized = (points + 1) / 2 * 256
        points_cam = converter(points_unnormalized)

        target_cam = converter(target)

        heatmap = to_heatmap(target[None], topdown[None]).squeeze()
        heatmap_cam = to_heatmap(target_cam[None], rgb[None]).squeeze()

        _heatmap = heatmap.cpu().squeeze().numpy() / 10.0 + 0.9
        _heatmap_cam = heatmap_cam.cpu().squeeze().numpy() / 10.0 + 0.9

        _rgb = (rgb.cpu() * 255).byte().numpy().transpose(1, 2, 0)[:, :, :3]
        _rgb[heatmap_cam > 0.1] = 255
        _rgb = Image.fromarray(_rgb)

        _topdown = common.COLOR[topdown.argmax(0).cpu().numpy()]
        _topdown[heatmap > 0.1] = 255
        _topdown = Image.fromarray(_topdown)
        _draw_map = ImageDraw.Draw(_topdown)
        _draw_rgb = ImageDraw.Draw(_rgb)

        for x, y in points_unnormalized:
            _draw_map.ellipse((x-2, y-2, x+2, y+2), (255, 0, 0))

        for x, y in converter.cam_to_map(points_cam):
            _draw_map.ellipse((x-1, y-1, x+1, y+1), (0, 255, 0))

        for x, y in points_cam:
            _draw_rgb.ellipse((x-2, y-2, x+2, y+2), (255, 0, 0))

        _topdown.thumbnail(_rgb.size)

        cv2.imshow('debug', cv2.cvtColor(np.hstack((_rgb, _topdown)), cv2.COLOR_BGR2RGB))
        cv2.waitKey(0)
