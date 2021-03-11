from pathlib import Path

import numpy as np
import torch
import imgaug.augmenters as iaa
import pandas as pd

from torch.utils.data import Dataset, IterableDataset, DataLoader
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


def get_dataset(dataset_dir, is_train=False, batch_size=4, num_workers=4, sample_by='none', n=0, **kwargs):
    data = list()
    transform = transforms.Compose([
        lambda x: x,
        #transforms.ToTensor()
        ])

    episodes = list(sorted(Path(dataset_dir).glob('*')))
    episodes = [elem for elem in episodes if '.yml' not in str(elem)]
    #print(episodes)

    for i, _dataset_dir in enumerate(episodes):
        add = False
        add |= (is_train and i % 10 < 9)
        add |= (not is_train and i % 10 >= 9)

        if add:
            data.append(CarlaDataset(_dataset_dir, transform, **kwargs, n=n))

    print('%d frames.' % sum(map(len, data)))

    weights = torch.DoubleTensor(get_weights(data, key=sample_by))
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
    data = torch.utils.data.ConcatDataset(data)

    return Wrap(data, sampler, batch_size, 1000 if is_train else 100, num_workers)

def preprocess_semantic(semantic_np):
    topdown = CONVERTER[semantic_np]
    topdown = torch.LongTensor(topdown)
    topdown = torch.nn.functional.one_hot(topdown, N_CLASSES).permute(2, 0, 1).float()

    return topdown

class CarlaDataset(Dataset):
    def __init__(self, replay_buffer, num_samples):
        self.buffer = replay_buffer
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, i):
        print(len(self.buffer.states))
        state, action, reward, done, next_state, info = self.buffer.sample(i)

        topdown, target = state
        topdown = Image.fromarray(topdown)
        topdown = topdown.crop((128, 0, 128 + 256, 256))
        topdown = preprocess_semantic(np.array(topdown))
        target = torch.FloatTensor(np.float32(target))
        action = torch.FloatTensor(np.float32(action)) # model predictions, not expert
        reward = torch.FloatTensor([reward])
        done = torch.FloatTensor([done])

        ntopdown, ntarget = next_state
        ntopdown = Image.fromarray(ntopdown)
        ntopdown = ntopdown.crop((128, 0, 128 + 256, 256))
        ntopdown = preprocess_semantic(np.array(ntopdown))
        ntarget = torch.FloatTensor(np.float32(ntarget))
    
        return (topdown, target), action, reward, (ntopdown, ntarget), done, info

def get_dataloader(buf, is_train=False, num_workers=4):
    num_samples = buf.buffer_size if is_train else buf.batch_size #
    weights = np.ones(num_samples) # prioritize?
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, num_samples)
    return DataLoader(CarlaDataset(buf, num_samples), batch_size=buf.batch_size, num_workers=num_workers, sampler=sampler, drop_last=True, pin_memory=True)


if __name__ == '__main__':
    import sys
    import cv2
    import argparse
    from PIL import ImageDraw
    from lbc.carla_project.src.utils.heatmap import ToHeatmap
    from rl.dspred.online_map_model import MapModel
    import pickle as pkl
    from itertools import repeat

    model = MapModel.load_from_checkpoint('/home/aaron/workspace/carla/leaderboard-devkit/team_code/rl/config/weights/map_model.ckpt')
    model.eval()
    model.cuda()

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    with open('buffer.pkl', 'rb') as f:
        buf = pkl.load(f)
    print(buf.buffer_size)
    print(len(buf.states))
    batch_size = 4
    loader = get_dataloader(buf, batch_size)
    print(len(loader))

    for j, batch in enumerate(loader):
        state, action, reward, next_state, done, info = batch
        #print(action)
        topdown, target = state
        ntopdown, ntarget = next_state
        with torch.no_grad():
            points, (vmap, hmap) = model.forward(topdown.cuda(), target.cuda(), debug=True)
            npoints, (nvmap, nhmap) = model.forward(ntopdown.cuda(), ntarget.cuda(), debug=True)
        points = (points + 1) / 2 * 256
        npoints =(npoints + 1) / 2 * 256

        for i in range(action.shape[0]):
            _topdown = COLOR[topdown[i].argmax(0).cpu()]
            #heatmap = to_heatmap(target[None], topdown[None]).squeeze()
            _topdown[hmap[i][0].cpu() > 0.1] = 255
            _topdown = Image.fromarray(_topdown)

            _draw = ImageDraw.Draw(_topdown)
            for x, y in points[i].cpu().numpy():
                _draw.ellipse((x-2, y-2, x+2, y+2), (0,255,0))
            _draw.text((5, 10), f'reward = {reward[i].item():.5f}', (255,255,255))
            _topdown = cv2.cvtColor(np.array(_topdown), cv2.COLOR_BGR2RGB)

            # next state
            _ntopdown = COLOR[ntopdown[i].argmax(0).cpu().numpy()]
            _ntopdown[nhmap[i][0].cpu() > 0.1] = 255
            _ntopdown = cv2.cvtColor(_ntopdown, cv2.COLOR_BGR2RGB)

            _combined = np.hstack((_topdown, _ntopdown))
            cv2.imshow(f'topdown {i}', _combined)
        cv2.waitKey(0)

