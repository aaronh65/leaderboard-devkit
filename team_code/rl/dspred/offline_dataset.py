from pathlib import Path

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


# running mean for efficient prioritized experience replay?
def get_dataset(data=None, batch_size=4, num_workers=4, sample_by='none', n=0, **kwargs):
    data = list()
    transform = transforms.Compose([
        lambda x: x,
        transforms.ToTensor()
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
    def __init__(self, dataset_dir, transform=transforms.ToTensor(), n=0, max_len=50000):
        dataset_dir = Path(dataset_dir)
        measurements = list(sorted((dataset_dir / 'measurements').glob('*.json')))

        self.transform = transform
        self.dataset_dir = dataset_dir
        self.frames = list()
        self.measurements = pd.DataFrame([eval(x.read_text()) for x in measurements])
        self.converter = Converter()

        # n-step returns
        self.n = n
        self.gamma = 0.99
        self.discount = [self.gamma**i for i in range(n+1)]

        print(dataset_dir)

        for image_path in sorted((dataset_dir / 'topdown').glob('*.png')):
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
        #print(path / 'topdown' / ('%s.png' % frame))
        topdown = Image.open(path / 'topdown' / ('%s.png' % frame))
        topdown = topdown.crop((128, 0, 128 + 256, 256))
        topdown = preprocess_semantic(np.array(topdown))
        tick_data = self.measurements.iloc[i]
        target = torch.FloatTensor(np.float32((tick_data['x_tgt'], tick_data['y_tgt'])))
        action = torch.FloatTensor(np.float32((tick_data['x_aim'], tick_data['y_aim'])))

        ni = i + self.n + 1
        ni = min(ni, len(self.frames))
        reward = self.measurements.loc[i:ni-1, 'reward'].to_numpy()
        reward = np.multiply(reward, self.discount[:ni-i]).sum()
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
    from rl.dspred.map_model import MapModel

    model = MapModel.load_from_checkpoint('/home/aaron/workspace/carla/leaderboard-devkit/team_code/rl/config/weights/map_model.ckpt')
    model.eval()

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--n', type=int, default=0)
    args = parser.parse_args()

    data = CarlaDataset(args.dataset_dir, n=args.n)
    #val_data = get_dataset(args.dataset_dir, False, 4, sample_by='none', n=args.n)
    converter = Converter()
    to_heatmap = ToHeatmap(5)

    data_let = len(val_data)
    batch = 
    for batch in val_data:
        state, action, reward, next_state, done = batch

        # this state
        topdown, target = state
        ntopdown, ntarget = next_state
        with torch.no_grad():
            points, (vmap, hmap) = model.forward(topdown, target, debug=True)
        points = (points + 1) / 2 * 256

        for i in range(action.shape[0]):
            _topdown = COLOR[topdown[i].argmax(0).cpu()]
            #heatmap = to_heatmap(target[None], topdown[None]).squeeze()
            _topdown[hmap[i][0].cpu() > 0.1] = 255
            _topdown = Image.fromarray(_topdown)

            _draw = ImageDraw.Draw(_topdown)
            x, y = action[i].cpu().squeeze().numpy().astype(np.uint8)
            _draw.ellipse((x-2, y-2, x+2, y+2), (0,255,0))
            for x, y in points[i].cpu().numpy():
                _draw.ellipse((x-2, y-2, x+2, y+2), (0,255,0))
            _draw.text((5, 10), f'reward = {reward[i].item():.5f}', (255,255,255))
            _topdown = cv2.cvtColor(np.array(_topdown), cv2.COLOR_BGR2RGB)

            # next state
            _ntopdown = COLOR[ntopdown[i].argmax(0).cpu().numpy()]
            nheatmap = to_heatmap(ntarget[i:i+1], ntopdown[i:i+1]).squeeze()
            _ntopdown[nheatmap > 0.1] = 255
            _ntopdown = cv2.cvtColor(_ntopdown, cv2.COLOR_BGR2RGB)

            _combined = np.hstack((_topdown, _ntopdown))
            cv2.imshow(f'topdown {i}', _combined)
        cv2.waitKey(0)

