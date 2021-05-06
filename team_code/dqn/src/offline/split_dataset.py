from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageDraw

from lbc.carla_project.src.converter import PIXELS_PER_WORLD
from lbc.carla_project.src.dataset_wrapper import Wrap
from lbc.carla_project.src.common import *
from misc.utils import *

# Reproducibility.
#np.random.seed(0)
#torch.manual_seed(0)

# Data has frame skip of 5.
GAP = 10
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

def get_dataloader(hparams, dataset_dir, is_train=True, shuffle=True, sample_by='even', **kwargs):
    dataset_dir = Path(dataset_dir) / 'data'

    episodes = list()

    routes = sorted(list(dataset_dir.glob('*')))
    #routes = sorted([path for path in routes if path.is_dir() and 'logs' not in str(path)])
    for route in routes:
        episodes.extend(sorted(route.glob('*')))
    dataset = SplitCarlaDataset(hparams, episodes, is_train)
    dataloader = DataLoader(
        dataset, 
        batch_size=hparams.batch_size, 
        num_workers=hparams.num_workers, 
        shuffle=shuffle, 
        pin_memory=True, 
        drop_last=True)
    return dataloader

def preprocess_semantic(semantic_np):
    topdown = CONVERTER[semantic_np]
    topdown = torch.LongTensor(topdown)
    topdown = torch.nn.functional.one_hot(topdown, N_CLASSES).permute(2, 0, 1).float()

    return topdown

class SplitCarlaDataset(Dataset):
    def __init__(self, hparams, episodes, is_train):

        # check dataset type
        if 'autopilot' in str(episodes[0]):
            self.expert_type = 'autopilot'
        elif 'privileged' in str(episodes[0]):
            self.expert_type = 'privileged'
        else:
            print('expert type not understood')
            raise Exception

        self.hparams = hparams
        self.transform = lambda x: x
        self.discount = [hparams.gamma**i for i in range(hparams.n+1)]

        self.dataset_root = episodes[0].parent.parent
        self.all_frames = list()
        self.hard_indices = list()
        self.bad_indices = set()
        self.measurements = pd.DataFrame()
        thresh = GAP*STEPS + hparams.n + 1
        infraction_count = 0
        for i, _dataset_dir in enumerate(episodes): # 90-10 train/val
            topdown_frames = list(sorted((Path(_dataset_dir) / 'topdown').glob('*')))
            dataset_len = len(topdown_frames)
            if dataset_len <= thresh:
                print(f'{_dataset_dir} invalid, skipping...')
                continue
            else:
                print(_dataset_dir)

            begin = len(self.all_frames)
            self.all_frames.extend(topdown_frames)
            end = len(self.all_frames)

            measurements = list(sorted((_dataset_dir / 'measurements').glob('*')))
            measurements = pd.DataFrame([eval(x.read_text()) for x in measurements])
            self.measurements = self.measurements.append(measurements, ignore_index=True)

            # recover bad indices (most restrictive for autopilot datasets)
            self.bad_indices.update(range(end-thresh, end))

            # recover hard indices
            rewards = measurements['reward'].to_numpy()
            hard_indices = begin + np.where(rewards < 0)[0]
            hard_starts = hard_indices - hparams.n
            hard_starts = np.clip(hard_starts, begin, end)
            for start, end in zip(hard_starts, hard_indices):
                self.hard_indices.extend(range(start, end+1))
            infraction_count += len(hard_indices)
            
        # check indices
        self.dataset_len = len(self.all_frames)
        self.all_indices = np.arange(self.dataset_len)
        assert len(self.bad_indices) < self.dataset_len, \
                'no valid examples'
        self.hard_indices = [i for i in self.hard_indices if i not in self.bad_indices]
        self.hard_indices = sorted(list(set(self.hard_indices)))
        assert len(self.hard_indices) > 0 or self.hparams.hard_prop == 0, \
                'no hard examples present'
        print('%d frames.' % self.dataset_len)
        print('%d infractions.' % infraction_count)
        print('%d hard frames.' % len(self.hard_indices))

        # n-step returns
        self.epoch_len = 1000 if is_train else 250
        self.epoch_len = self.epoch_len * hparams.batch_size

        # step count
        #self.register_buffer('dataset_step', torch.Tensor[0])
        self.dataset_step = 0


        if self.hparams.max_prop_epoch is None:
            self.max_prop_epoch = self.hparams.max_epochs // 2
        else:
            self.max_prop_epoch = self.hparams.max_prop_epoch

        self.epoch_num = 0
        #print('is_train', is_train, self.max_prop_batches)
        #if hasattr(self.hparams, 'direct_indexing'):
        #    self.direct_indexing = self.hparams.direct_indexing
        #else:
        #    self.direct_indexing = False


    def __len__(self):
        return self.epoch_len

    def __getitem__(self, i):
        if self.dataset_step == 0:
            np.random.seed()

        ## retrieving mixed batches
        frac = 1 if self.max_prop_epoch == 0 else min(self.epoch_num / self.max_prop_epoch, 1)
        hard_prop = self.hparams.hard_prop * frac
        indices = self.hard_indices if np.random.random() < hard_prop else self.all_indices
        count = 0
        while True:
            i = np.random.choice(indices)
            if i not in self.bad_indices:
                break
            count += 1
            if count > 999:
                print('could not valid index')
                raise Exception

        topdown_frame = self.all_frames[i]
        route, rep, _, frame = topdown_frame.parts[-4:]
        frame = frame[:-4]
        meta = '%s/%s/%s' % (route, rep, frame)
        
        # check for n-step return end index
        ni = i + self.hparams.n + 1
        ni = min(ni, len(self.all_frames)-1)
        # check if episode terminated prematurely, pd.df is end-inclusive slicing
        done_list = self.measurements.loc[i:ni-1, 'done'].to_numpy().tolist() 
        done = int(1 in done_list)
        if done:
            ni = i + done_list.index(1) + 1
        done = torch.FloatTensor(np.float32([done]))
        
        # reward calculations
        penalty = self.measurements.loc[i:ni-1, 'infraction_penalty'].to_numpy()
        route_reward = self.measurements.loc[i:ni-1, 'route_reward'].to_numpy()
        discounts = self.discount[:len(penalty)]
        discount = torch.FloatTensor(np.float32([self.discount[ni-i-1]])) # for Q(ns)
        reward = route_reward - penalty
        reward = np.dot(reward, discounts) # discounted sum of rewards

        # transform to log scale (DQfD)
        sign = -1 if reward < 0 else 1
        reward = sign * np.log(1+np.abs(reward))
        reward = torch.FloatTensor(np.float32([reward]))

        # turn off margin if expert action is bad?
        margin_switch = 0 if reward < 0 else 1
        margin_switch = torch.FloatTensor(np.float32([margin_switch]))
        
        # topdown, target, points
        topdown = Image.open(topdown_frame)
        topdown = topdown.crop((128,0,128+256,256))
        topdown = preprocess_semantic(np.array(topdown))

        tick_data = self.measurements.iloc[i]
        target = torch.FloatTensor(np.float32((tick_data['x_target'], tick_data['y_target'])))
        action = self._get_action(tick_data, frame, i)
        
        # next state, next target
        ntopdown_frame = self.all_frames[ni]
        _, _, _, nframe = ntopdown_frame.parts[-4:]
        nframe = nframe[:-4]

        ntopdown = Image.open(ntopdown_frame)
        ntopdown = ntopdown.crop((128,0,128+256,256))
        ntopdown = preprocess_semantic(np.array(ntopdown))

        ntick_data = self.measurements.iloc[ni]
        ntarget = torch.FloatTensor(np.float32((ntick_data['x_target'], ntick_data['y_target'])))
        naction = self._get_action(ntick_data, nframe, ni)

        info = {
            'discount': discount, 
            'naction': naction,
            'hard_prop': torch.Tensor([hard_prop]),
            'metadata': torch.Tensor(encode_str(meta)),
            'margin_switch': margin_switch
        }


        self.dataset_step += 1
        # state, action, reward, next_state, done, info
        return (topdown, target), action, reward, (ntopdown, ntarget), done, info


    def _get_action(self, tick_data, frame, i):
        points_expert = self._get_points(frame, i)
        if hasattr(self.hparams, 'throttle_mode'):
            control_expert = self._get_control(tick_data)
            action = (points_expert, control_expert)
        else:
            action = points_expert
        return action

    def _get_control(self, tick_data):
        steer, throttle, target_speed = tick_data[['steer', 'throttle', 'target_speed']]
        if self.hparams.throttle_mode == 'speed':
            target_speed = 0 if tick_data['brake'] else target_speed
            target_speed = min(target_speed / self.hparams.max_speed, 1.0)
            control_expert = np.float32([steer, target_speed])
        elif self.hparams.throttle_mode == 'throttle':
            throttle = 0 if tick_data['brake'] else throttle
            control_expert = np.float32([steer, throttle])
        else:
            raise Exception
        return control_expert


    def _get_points(self, frame, i):
        if self.expert_type == 'autopilot':
            u = np.float32(self.measurements.iloc[i][['x_position', 'y_position']])
            theta = self.measurements.iloc[i]['theta']
            R = np.array([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta),  np.cos(theta)],
                ])

            points = list()
            for skip in range(1, STEPS+1):
                j = i + GAP * skip
                try:
                    v = np.array(self.measurements.iloc[j][['x_position', 'y_position']])
                except Exception:
                    print('ILLEGAL INDEX FOUND')
                    print('start', i)
                    print('GAP', GAP)
                    print('skip', skip)
                    print('j', j)
                    sys.exit(1)

                target = R.T.dot(v - u)
                target *= PIXELS_PER_WORLD
                target += [128, 256]

                points.append(target)

            points = torch.FloatTensor(points)
            points = torch.clamp(points, 0, 256)
            points = (points / 256) * 2 - 1
        elif self.expert_type == 'privileged':
            topdown_frame = self.all_frames[i]
            route, rep, _, frame = topdown_frame.parts[-4:]
            frame = frame[:-4]
            with open(self.dataset_root / route / rep / 'points_lbc' / f'{frame}.npy', 'rb') as f:
                points = np.load(f)
            points = (points / 256) * 2 - 1
        else:
            print('unknown expert type')
            raise Exception
        return points
       
    

if __name__ == '__main__':

    import cv2, argparse
    import matplotlib as mpl
    mpl.use('TkAgg')
    import matplotlib.pyplot as plt
    from dqn.src.agents.map_model import MapModel
    from dqn.src.agents.heatmap import ToHeatmap
    
    to_heatmap = ToHeatmap(5)
    parser = argparse.ArgumentParser()
    #parser.add_argument('--dataset_dir', type=Path,
    #        default='/data/aaronhua/leaderboard/data/lbc/autopilot/autopilot_devtest_toy')
    parser.add_argument('--dataset_dir', type=Path,
            default='/data/aaronhua/leaderboard/data/lbc/privileged_agent/privileged_devtest_toy')

    #parser.add_argument('--dataset_dir', type=Path, 
    #        default='/data/aaronhua/leaderboard/data/lbc/autopilot/autopilot_devtest')
    #parser.add_argument('--weights_path', type=str,
    #        default='/data/aaronhua/leaderboard/training/lbc/20210405_225046/epoch=22.ckpt')
    parser.add_argument('--n', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_visuals', action='store_true')
    parser.add_argument('--throttle_mode', type=str, default='speed')
    parser.add_argument('--max_speed', type=int, default=10)
    parser.add_argument('--hard_prop', type=float, default=0.0)
    parser.add_argument('--max_prop_epoch', type=float, default=0.0)
    parser.add_argument('--max_epochs', type=int, default=10)

    args = parser.parse_args()
    loader = get_dataloader(args, args.dataset_dir, is_train=True)
    to_heatmap = ToHeatmap(5)

    for epoch in range(args.max_epochs):
        for batch_nb, batch in enumerate(loader):
            state, action, reward, next_state, done, info = batch
            topdown, target = state
            points, ctrl = action
            points = (points + 1)/2 * 256
            points = points[0]

            tmap = to_heatmap(target, topdown)
            topdown = COLOR[topdown.argmax(1).cpu()]
            topdown[tmap > 0.1] = 255
            topdown = topdown[0]
            topdown = Image.fromarray(topdown)
            draw = ImageDraw.Draw(topdown)
            for x,y in points:
                draw.ellipse((x-2,y-2,x+2,y+2), (255,0,0))
            topdown = cv2.cvtColor(np.array(topdown), cv2.COLOR_RGB2BGR)
            cv2.imshow('topdown', topdown)
            cv2.waitKey(100)
