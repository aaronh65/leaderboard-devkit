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
def get_dataloader(hparams, is_train=False):

    mode = hparams.data_mode
    if mode == 'hybrid' or mode == 'online':
        assert hparams.num_workers == 0, 'cannot use non-stationary datasets with multiple workers'

    
    # retrieve data paths
    if mode == 'offline':
        assert hasattr(hparams, 'dataset_dir'), 'no offline dataset location'
        episodes = list()
        routes = Path(hparams.dataset_dir).glob('*')
        routes = sorted([path for path in routes if path.is_dir() and 'logs' not in str(path)])
        for route in routes:
            episodes.extend(sorted(route.glob('*')))

        # data augmentations
        #transform = transforms.Compose([
        #    torchvision.transforms.RandomAffine(degrees=) if is_train else lambda x: x,
        #    transforms.ToTensor()
        #    ])
        # need at least 10 episode directories for split data = list()

        data = list()
        for i, _dataset_dir in enumerate(episodes): # 90-10 train/val
            add = False
            add |= (is_train and i % 10 < 9)
            add |= (not is_train and i % 10 >= 9)

            if add:
                data.append(OfflineCarlaDataset(hparams, _dataset_dir))

    ## sum up the lengths of each dataset

    data_len = sum(map(len, data))
    num_samples = data_len // hparams.batch_size
    print('train set' if is_train else 'val_set')
    print(f'data_len: {data_len}')
    print(f'batch_size: {hparams.batch_size}')
    print(f'epoch_len: {num_samples}')

    num_samples=1000 if is_train else 100
    weights = torch.DoubleTensor(get_weights(data, key='none'))
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
    #sampler = None
    data = torch.utils.data.ConcatDataset(data)

    # fourth argument specifies steps/batches per epoch
    return Wrap(data, sampler, hparams.batch_size, num_samples, hparams.num_workers)

def preprocess_semantic(semantic_np):
    topdown = CONVERTER[semantic_np]
    topdown = torch.LongTensor(topdown)
    topdown = torch.nn.functional.one_hot(topdown, N_CLASSES).permute(2, 0, 1).float()

    return topdown


class OfflineCarlaDataset(Dataset):
    def __init__(self, hparams, dataset_dir, transform=transforms.ToTensor()):
        dataset_dir = Path(dataset_dir)
        measurements = list(sorted((dataset_dir / 'measurements').glob('*.json')))

        print(dataset_dir)

        self.hparams = hparams
        self.transform = transform
        self.dataset_dir = dataset_dir
        self.measurements = pd.DataFrame([eval(x.read_text()) for x in measurements])

        self.pixel_jitter = self.hparams.pixel_jitter
        self.angle_jitter = self.hparams.angle_jitter

        # n-step returns
        self.discount = [self.hparams.gamma**i for i in range(hparams.n+1)]

        self.frames = list()
        for image_path in sorted((dataset_dir / 'topdown').glob('*.png')):
            frame = str(image_path.stem)

            assert (dataset_dir / 'topdown' / ('%s.png' % frame)).exists()
            assert int(frame) < len(self.measurements)

            self.frames.append(frame)

        assert len(self.frames) > 0, '%s has 0 frames.' % dataset_dir

    def __len__(self):
        return len(self.frames)

    # topdown (C,H,W) 
    # points (K,4,2) where K is # of point sets e.g. points_lbc, points_dqn
    def augment_and_crop(self, topdown, ntopdown, points):
        dr = np.random.randint(-self.angle_jitter, self.angle_jitter+1)

        pixel_rotation = cv2.getRotationMatrix2D((256//2,256), dr, 1) # 2x3
        points_dqn, points_lbc = points
        points_dqn = np.hstack((points_dqn, np.ones((4,1)))) # 4x3
        points_dqn = np.matmul(pixel_rotation, points_dqn.T).T # 4x2
        points_lbc = np.hstack((points_lbc, np.ones((4,1)))) # 4x3
        points_lbc = np.matmul(pixel_rotation, points_lbc.T).T # 4x2


        image_rotation = cv2.getRotationMatrix2D((256,256), dr, 1) # 2x3

        topdown = np.array(topdown)
        topdown = cv2.warpAffine(topdown, image_rotation, 
                topdown.shape[1::-1], 
                flags=cv2.INTER_NEAREST)
        topdown = Image.fromarray(topdown)
        topdown = topdown.crop((128,0,128+256,256))
        topdown = preprocess_semantic(np.array(topdown))

        ntopdown = np.array(ntopdown)
        ntopdown = cv2.warpAffine(ntopdown, image_rotation, 
                ntopdown.shape[1::-1], 
                flags=cv2.INTER_NEAREST)
        ntopdown = Image.fromarray(ntopdown)
        ntopdown = ntopdown.crop((128,0,128+256,256))
        ntopdown = preprocess_semantic(np.array(ntopdown))


        #dx = np.random.randint(-self.pixel_jitter, self.pixel_jitter+1)
        #dy = np.random.randint(0, self.pixel_jitter+1) - pixel_offset

        return topdown, ntopdown, (points_dqn, points_lbc)

    def __getitem__(self, i):
        path = self.dataset_dir
        frame = self.frames[i]
        ni = i + self.hparams.n + 1 # index of the next state
        ni = min(ni, len(self.frames)-1)
        nframe = self.frames[ni]

        with open(path / 'points_dqn' / f'{frame}.npy', 'rb') as f:
            points_dqn = np.load(f)
        with open(path / 'points_lbc' / f'{frame}.npy', 'rb') as f:
            points_lbc = np.load(f)

        topdown = Image.open(path / 'topdown' / ('%s.png' % frame))
        ntopdown = Image.open(path / 'topdown' / ('%s.png' % nframe))
        topdown, ntopdown, (points_dqn, points_lbc) = self.augment_and_crop(
                topdown, ntopdown, (points_dqn, points_lbc))

        tick_data = self.measurements.iloc[i]
        target = torch.FloatTensor(np.float32((tick_data['x_tgt'], tick_data['y_tgt'])))

        # panda df slicing is end index inclusive
        done = self.measurements.loc[i:ni-1, 'done'].to_numpy()
        if 1 in done:
            ni = i + done.index(1) + 1
            done = 1
        else:
            done = 0
        done = torch.FloatTensor(np.float32([done]))

        # reward
        penalty = self.measurements.loc[i:ni-1, 'penalty'].to_numpy()
        imitation_reward = self.measurements.loc[i:ni-1, 'imitation_reward'].to_numpy()
        route_reward = self.measurements.loc[i:ni-1, 'route_reward'].to_numpy()
        discounts = self.discount[:len(penalty)]

        reward = route_reward - penalty
        reward = np.dot(reward, discounts)
        reward = torch.FloatTensor(np.float32([reward]))

        ntick_data = self.measurements.iloc[ni]
        ntarget = torch.FloatTensor(np.float32((ntick_data['x_tgt'], ntick_data['y_tgt'])))

                
        # last discount
        discount = torch.FloatTensor(np.float32([self.discount[ni-i-1]]))

        info = {'discount': discount, 
                'points_dqn': points_dqn, 
                'points_lbc': points_lbc,
                #'debug_path': debug_path
                }

        # state, action, reward, next_state, done, info
        return (topdown, target), points_lbc, reward, (ntopdown, ntarget), done, info
        


if __name__ == '__main__':
    import os, cv2, argparse
    from PIL import ImageDraw
    from rl.dspred.map_model import MapModel, visualize
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=Path)
    parser.add_argument('--n', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--mode', type=str, default='offline', 
            choices=['offline', 'online', 'hybrid'])
    parser.add_argument('--angle_jitter', type=float, default=15)
    parser.add_argument('--pixel_jitter', type=int, default=15)
    args = parser.parse_args()

    if args.dataset_dir is None: # local
        #args.dataset_dir = '/data/leaderboard/data/rl/dspred/20210311_213726'
        args.dataset_dir = '/data/leaderboard/data/rl/dspred/debug/20210311_143718'

    project_root = os.environ['PROJECT_ROOT']
    RESUME = f'{project_root}/team_code/rl/config/weights/lbc_expert.ckpt'
    model = MapModel.load_from_checkpoint(RESUME)
    model.cuda()
    model.eval()

    model.hparams.dataset_dir = args.dataset_dir
    model.hparams.batch_size = args.batch_size
    #model.hparams.save_dir = args.save_dir
    model.hparams.n = args.n
    model.hparams.data_mode = args.mode
    model.hparams.gamma = args.gamma
    model.hparams.num_workers = args.num_workers
    model.hparams.angle_jitter = args.angle_jitter
    model.hparams.pixel_jitter = args.pixel_jitter

    converter = Converter()
    loader = get_dataloader(model.hparams, is_train=True)

    for batch in loader:

        state, action, reward, next_state, done, info = batch
        action = action.cuda()
        reward = reward.cuda()
        done = done.cuda()

        if True:
            
            # this state
            topdown, target = state
            with torch.no_grad():
                points, vmap, hmap = model.forward(topdown.cuda(), target.cuda(), debug=True)
            Q_all = model.get_Q_values(vmap, action)
            Q = torch.mean(Q_all[:, :2], axis=1, keepdim=False)

            ntopdown, ntarget = next_state
            with torch.no_grad():
                npoints, nvmap, nhmap = model.forward(ntopdown.cuda(), ntarget.cuda(), debug=True)
            naction, nQ_all = model.get_dqn_actions(nvmap)
            nQ = torch.mean(nQ_all[:,:2], axis=1, keepdim=False)

            # td loss
            discount = info['discount'].cuda()
            td_target = reward + discount * nQ * (1-done)
            td_loss = model.td_criterion(Q, td_target) # TD(n) error Nx1

            # expert margin loss
            expert_heatmap = model.expert_heatmap(info['points_lbc'], vmap)
            margin = vmap + (1 - expert_heatmap)
            _, Q_margin = model.get_dqn_actions(margin)
            Q_expert = model.get_Q_values(vmap, info['points_lbc'].cuda())
            margin_loss = Q_margin - Q_expert # Nx4
            margin_loss = model.margin_weight*torch.mean(margin_loss, axis=1, keepdim=False) # Nx1

            meta = {
                    'Q': Q, 'nQ': nQ, 'hparams': model.hparams,
                    'td_loss': td_loss, 
                    'margin_loss': margin_loss,
                    }


            visualize(batch, vmap, hmap, nvmap, nhmap, naction, meta)
        cv2.waitKey(0)
        break
