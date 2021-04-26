from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch

from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from lbc.carla_project.src.converter import PIXELS_PER_WORLD
from lbc.carla_project.src.dataset_wrapper import Wrap
from lbc.carla_project.src.common import *
from misc.utils import *

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

def get_dataloader(hparams, dataset_dir, is_train=False):
    dataset_dir = Path(dataset_dir) / 'data'

    episodes = list()

    routes = sorted(list(dataset_dir.glob('*')))
    #routes = sorted([path for path in routes if path.is_dir() and 'logs' not in str(path)])
    for route in routes:
        episodes.extend(sorted(route.glob('*')))
    dataset = CarlaDataset(hparams, episodes, is_train)
    dataloader = DataLoader(
            dataset, 
            batch_size=hparams.batch_size, 
            num_workers=hparams.num_workers, 
            #shuffle=True, 
            pin_memory=True, 
            drop_last=True)
    return dataloader


def preprocess_semantic(semantic_np):
    topdown = CONVERTER[semantic_np]
    topdown = torch.LongTensor(topdown)
    topdown = torch.nn.functional.one_hot(topdown, N_CLASSES).permute(2, 0, 1).float()

    return topdown


class CarlaDataset(Dataset):
    def __init__(self, hparams, episodes, is_train):
        self.hparams = hparams
        self.transform = lambda x: x
        self.discount = [hparams.gamma**i for i in range(hparams.n+1)]

        self.topdown_frames = []
        self.measurements = pd.DataFrame()
        for i, _dataset_dir in enumerate(episodes): # 90-10 train/val
            topdown_frames = list(sorted((Path(_dataset_dir) / 'topdown').glob('*')))
            dataset_len = len(topdown_frames)
            if dataset_len <= GAP*STEPS:
                print(f'{_dataset_dir} invalid, skipping...')
                continue
            else:
                print(_dataset_dir)
            self.topdown_frames.extend(topdown_frames)
            measure_frames = list(sorted((_dataset_dir / 'measurements').glob('*')))
            measurements = pd.DataFrame([eval(x.read_text()) for x in measure_frames])
            self.measurements = self.measurements.append(measurements, ignore_index=True)

        self.dataset_len = len(self.topdown_frames) - hparams.n - hparams.batch_size
        print('%d frames.' % self.dataset_len)

        # n-step returns
        self.epoch_len = 1000 if is_train else 250
        self.epoch_len = self.epoch_len * hparams.batch_size

        weights = np.ones(self.dataset_len)*10
        prefix = 'train' if is_train else 'val'
        self.weight_path = hparams.save_dir / f'{prefix}_losses.npy'
        #self.weight_path.mkdir(exist_ok=True, parents=True)
        with open(self.weight_path, 'wb') as f:
            np.save(f, weights)

        self.step = 0
        self.base_update_rate = 100
        self.weight_update_rate = self.base_update_rate * hparams.batch_size // hparams.num_workers
        #self.beta = 1e-2


        infractions = set(self.measurements['infraction'].to_numpy().tolist())
        print(infractions)

    def __len__(self):
        #return self.epoch_len
        return self.dataset_len

    def _recompute_weights(self):

        # buffer time in case map model is still saving weights
        with open(self.weight_path, 'rb') as f:
            weights = np.load(f)

        #print(f'recomputing weights at dataset step {self.step}, {sum(weights !=10)} changed elements')
        weights = np.clip(weights, 0, 10)
        weights = np.exp(weights)
        weights = weights / np.sum(weights)
        #weights = weights + eps
        #weights /= np.sum(weights)
        #weights /= np.sum(weights)

        # regen weights
        np.random.seed()
        self.indices = np.random.choice(
                np.arange(self.dataset_len), size=self.epoch_len, p=weights)
        #self.beta = min(self.beta + 1e-2, 1)

    def __getitem__(self, i):
        #print('dataset step', self.step)
        if self.step % self.weight_update_rate == 50 or self.step == 0:
            self._recompute_weights()

        i = self.indices[i]

        #path = self.dataset_dir
        topdown_frame = self.topdown_frames[i]
        route, rep, _, frame = topdown_frame.parts[-4:]
        frame = frame[:-4]
        meta = '%s/%s/%s' % (route, rep, frame)
        
        # check for n-step return end index
        ni = i + self.hparams.n + 1
        ni = min(ni, len(self.topdown_frames)-1)
        # check if episode terminated prematurely
        done_list = self.measurements.loc[i:ni-1, 'done'].to_numpy().tolist() # slicing end-inclusive
        done = int(1 in done_list)
        if done:
            ni = i + done_list.index(1) + 1
        done = torch.FloatTensor(np.float32([done]))
        
        # reward calculations
        penalty = self.measurements.loc[i:ni-1, 'penalty'].to_numpy()
        imitation_reward = self.measurements.loc[i:ni-1, 'imitation_reward'].to_numpy()
        route_reward = self.measurements.loc[i:ni-1, 'route_reward'].to_numpy()
        discounts = self.discount[:len(penalty)]
        discount = torch.FloatTensor(np.float32([self.discount[ni-i-1]])) # for Q(ns)

        reward = route_reward - penalty
        reward = np.dot(reward, discounts) # discounted sum of rewards
        margin_switch = 0 if reward < 0 else 1
        margin_switch = torch.FloatTensor(np.float32([margin_switch]))
        reward = torch.FloatTensor(np.float32([reward]))

        # topdown, target, points
        topdown = Image.open(topdown_frame)
        topdown = topdown.crop((128,0,128+256,256))
        topdown = preprocess_semantic(np.array(topdown))

        tick_data = self.measurements.iloc[i]
        target = torch.FloatTensor(np.float32((tick_data['x_tgt'], tick_data['y_tgt'])))
        path = topdown_frame.parent.parent
        with open(path / 'points_student' / f'{frame}.npy', 'rb') as f:
            points_student = torch.clamp(torch.Tensor(np.load(f)),0,255)
        with open(path / 'points_expert' / f'{frame}.npy', 'rb') as f:
            points_expert = torch.clamp(torch.Tensor(np.load(f)),0,255)

        # next state, next target
        ntopdown_frame = self.topdown_frames[ni]
        ntopdown = Image.open(ntopdown_frame)
        ntopdown = ntopdown.crop((128,0,128+256,256))
        ntopdown = preprocess_semantic(np.array(ntopdown))

        ntick_data = self.measurements.iloc[ni]
        ntarget = torch.FloatTensor(np.float32((ntick_data['x_tgt'], ntick_data['y_tgt'])))
        itensor = torch.FloatTensor(np.float32([i]))
        info = {'discount': discount, 
                'points_student': points_student,
                'points_expert': points_expert,
                'metadata': torch.Tensor(encode_str(meta)),
                'data_index': itensor,
                'margin_switch': margin_switch
                }
        self.step += 1 


        if tick_data['infraction'] != 'none':
            print(tick_data['infraction'])

        # state, action, reward, next_state, done, info
        return (topdown, target), points_student, reward, (ntopdown, ntarget), done, info
        

if __name__ == '__main__':
    import cv2, argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str,
            #default='/data/aaronhua/leaderboard/data/dqn/20210420_111906'
            default='/data/aaronhua/leaderboard/data/dqn/20210407_024101'
            )
    parser.add_argument('--n', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--train_dataset', type=Path)
    parser.add_argument('--val_dataset', type=Path)
    parser.add_argument('--id', type=str, default=datetime.now().strftime("%Y%m%d_%H%M%S")) 
    parser.add_argument('--save_dir', type=Path)
    parser.add_argument('--data_root', type=Path, default='/data/aaronhua')

    args = parser.parse_args()

    suffix = f'debug/{args.id}' 
    save_dir = args.data_root / f'leaderboard/training/dqn/offline/{suffix}'
    args.save_dir = save_dir
    args.save_dir.mkdir(parents=True, exist_ok=True)
    
    if args.train_dataset is None:
        args.train_dataset = Path('/data/aaronhua/leaderboard/data/dqn/20210407_024101')
    if args.val_dataset is None:
        args.val_dataset = Path('/data/aaronhua/leaderboard/data/dqn/20210420_111906')

    dataloader = get_dataloader(args, args.train_dataset, is_train=False)
    print(len(dataloader))

    for batch_nb, batch in tqdm(enumerate(dataloader)):
        break

