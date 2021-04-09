from pathlib import Path

import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

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

def get_dataloader(hparams, is_train=True, sample_by='even', **kwargs):
    data = list()

    #transform = transforms.Compose([
    #    get_augmenter() if is_train else lambda x: x,
    #    transforms.ToTensor()
    #    ])
    transform = lambda x:x

    episodes = list()
    dataset_dir = Path(hparams.dataset_dir) / 'data'
    routes = dataset_dir.glob('*')
    routes = sorted([path for path in routes if path.is_dir() and 'logs' not in str(path)])
    for route in routes:
        episodes.extend(sorted(route.glob('*')))

    data = list()
    for i, _dataset_dir in enumerate(episodes): # 90-10 train/val
        dataset_len = len(list((Path(_dataset_dir) / 'topdown').glob('*')))
        if dataset_len <= GAP*STEPS:
            print(f'{_dataset_dir} invalid, skipping...')
            continue
        add = False
        add |= (is_train and i % 10 < 9)
        add |= (not is_train and i % 10 >= 9)
        if add:
            data.append(CarlaDataset(_dataset_dir, hparams))
    
    print('%d frames.' % sum(map(len, data)))

    weights = torch.DoubleTensor(get_weights(data, key=sample_by))
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
    data = torch.utils.data.ConcatDataset(data)

    return Wrap(data, sampler, hparams.batch_size, 1000 if is_train else 100, hparams.num_workers)


def preprocess_semantic(semantic_np):
    topdown = CONVERTER[semantic_np]
    topdown = torch.LongTensor(topdown)
    topdown = torch.nn.functional.one_hot(topdown, N_CLASSES).permute(2, 0, 1).float()

    return topdown


class CarlaDataset(Dataset):
    def __init__(self, dataset_dir, hparams, transform=transforms.ToTensor()):
        dataset_dir = Path(dataset_dir)
        self.dataset_dir = dataset_dir
        measurements = list(sorted((dataset_dir / 'measurements').glob('*.json')))

        print(dataset_dir)

        self.hparams = hparams
        self.transform = transform
        self.measurements = pd.DataFrame([eval(x.read_text()) for x in measurements])

        #self.pixel_jitter = self.hparams.pixel_jitter
        #self.angle_jitter = self.hparams.angle_jitter

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
    # points (K,4,2) where K is # of point sets e.g. points_expert, points_student
    def augment_and_crop(self, topdown, ntopdown, points):
        
        dr = np.random.randint(-self.angle_jitter, self.angle_jitter+1)
        dx = np.random.randint(-self.pixel_jitter, self.pixel_jitter+1)
        dy = np.random.randint(0, self.pixel_jitter+1)

        points_student, points_expert = points
        offset = np.ones(points_student.shape)
        offset.T[0] = dx
        offset.T[1] = dy
        pixel_rotation = cv2.getRotationMatrix2D((256//2,256), dr, 1) # 2x3

        points_student = np.hstack((points_student, np.ones((4,1)))) # 4x3
        points_student = np.matmul(pixel_rotation, points_student.T).T # 4x2
        points_student = np.clip(points_student - offset, 0, 255)
        points_expert = np.hstack((points_expert, np.ones((4,1)))) # 4x3
        points_expert = np.matmul(pixel_rotation, points_expert.T).T # 4x2
        points_expert = np.clip(points_expert - offset, 0, 255)
        
        image_rotation = cv2.getRotationMatrix2D((256,256), dr, 1) # 2x3

        topdown = np.array(topdown)
        topdown = cv2.warpAffine(topdown, image_rotation, 
                topdown.shape[1::-1], flags=cv2.INTER_NEAREST)
        topdown = Image.fromarray(topdown)
        topdown = topdown.crop((128+dx,0+dy,128+256+dx,256+dy))
        topdown = preprocess_semantic(np.array(topdown))

        ntopdown = np.array(ntopdown)
        ntopdown = cv2.warpAffine(ntopdown, image_rotation, 
                ntopdown.shape[1::-1], flags=cv2.INTER_NEAREST)
        ntopdown = Image.fromarray(ntopdown)
        ntopdown = ntopdown.crop((128+dx,0+dy,128+256+dx,256+dy))
        ntopdown = preprocess_semantic(np.array(ntopdown))

        
        return topdown, ntopdown, (points_student, points_expert)

    def __getitem__(self, i):
        path = self.dataset_dir
        route, rep = path.parts[-2:]
        frame = self.frames[i]
        meta = '%s %s %s' % (route, rep, frame)
        
        # check if we're done in the next n steps
        ni = i + self.hparams.n + 1 # index of the next state in DQN
        ni = min(ni, len(self.frames)-1)

        # panda df slicing is end index inclusive
        done_list = self.measurements.loc[i:ni-1, 'done'].to_numpy()
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
        reward = torch.FloatTensor(np.float32([reward]))

        # topdown, target, points
        frame = self.frames[i]
        topdown = Image.open(path / 'topdown' / ('%s.png' % frame))
        topdown = topdown.crop((128,0,128+256,256))
        topdown = preprocess_semantic(np.array(topdown))

        tick_data = self.measurements.iloc[i]
        target = torch.FloatTensor(np.float32((tick_data['x_tgt'], tick_data['y_tgt'])))
        with open(path / 'points_student' / f'{frame}.npy', 'rb') as f:
            points_student = torch.clamp(torch.Tensor(np.load(f)),0,255)
        with open(path / 'points_expert' / f'{frame}.npy', 'rb') as f:
            points_expert = torch.clamp(torch.Tensor(np.load(f)),0,255)

        # next state, next target
        nframe = self.frames[ni]
        ntopdown = Image.open(path / 'topdown' / ('%s.png' % nframe))
        ntopdown = ntopdown.crop((128,0,128+256,256))
        ntopdown = preprocess_semantic(np.array(ntopdown))

        ntick_data = self.measurements.iloc[ni]
        ntarget = torch.FloatTensor(np.float32((ntick_data['x_tgt'], ntick_data['y_tgt'])))

        # LBC data augs
        #topdown, ntopdown, (points_student, points_expert) = self.augment_and_crop(
                #topdown, ntopdown, (points_student, points_expert))

        info = {'discount': discount, 
                'points_expert': points_expert,
                'meta': encode_str(meta)
                }

        
        # state, action, reward, next_state, done, info
        return (topdown, target), points_student, reward, (ntopdown, ntarget), done, info
        

if __name__ == '__main__':
    import cv2, argparse
    from dqn.src.agents.map_model import MapModel, visualize
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str,
            default='/data/aaronhua/leaderboard/data/dqn/20210407_024101')
    parser.add_argument('--weights_path', type=str,
            default='/data/aaronhua/leaderboard/training/lbc/20210405_225046/epoch=22.ckpt')
    parser.add_argument('--n', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--save_visuals', action='store_true')
    #parser.add_argument('--mode', type=str, default='offline', 
    #        choices=['offline', 'online', 'hybrid'])
    #parser.add_argument('--angle_jitter', type=float, default=45)
    #parser.add_argument('--pixel_jitter', type=int, default=16.5) # 3 meters

    args = parser.parse_args()

    if 'lbc' in args.weights_path:
        model = MapModel()
        model.restore_from_lbc(args.weights_path)
        model.hparams = args
    else:
        model = MapModel.load_from_checkpoint(args.weights_path)
    model.cuda()
    model.eval()

    model.hparams.dataset_dir = args.dataset_dir
    model.hparams.batch_size = args.batch_size
    model.hparams.n = args.n
    model.hparams.gamma = args.gamma
    model.hparams.num_workers = args.num_workers
    #model.hparams.angle_jitter = args.angle_jitter
    #model.hparams.pixel_jitter = args.pixel_jitter

    loader = get_dataloader(model.hparams, is_train=False)

    for i, batch in enumerate(loader):
        print(i)

        state, action, reward, next_state, done, info = batch
        action = action.cuda()
        reward = reward.cuda()
        done = done.cuda()

            
        # this state
        topdown, target = state
        with torch.no_grad():
            points, logits, weights, tmap = model.forward(topdown.cuda(), target.cuda(), debug=True)
        Q_all = model.get_Q_values(logits, action)
        #Q = torch.mean(Q_all[:, :2], axis=1, keepdim=False)
        Q = torch.mean(Q_all, axis=1, keepdim=False)

        ntopdown, ntarget = next_state
        with torch.no_grad():
            npoints, nlogits, nweights, ntmap = model.forward(ntopdown.cuda(), ntarget.cuda(), debug=True)
        naction, nQ_all = model.get_dqn_actions(nlogits)
        nQ = torch.mean(nQ_all, axis=1, keepdim=False)

        # td loss
        discount = info['discount'].cuda()
        td_target = reward + discount * nQ * (1-done)
        td_loss = model.td_criterion(Q, td_target) # TD(n) error Nx1

        # expert margin loss
        margin_map = torch.ones_like(logits)*1 # N,C,H,W
        points_expert = info['points_expert']
        margin_map[points_expert.long()] = 0
        #print(logits.shape)
        #print(points_expert.shape)
        #print(torch.min(points_expert,dim=-1)[0])
        #print(torch.max(points_expert,dim=-1)[0])
        Q_expert = model.get_Q_values(logits, points_expert.cuda())
        #Q_expert = model.get_Q_values(logits, points_expert)
        margin = logits + margin_map - Q_expert
        margin_loss = model.margin_weight*torch.mean(margin)

        #expert_heatmap = model.expert_heatmap(info['points_expert'], logits)
        #margin = logits + (1 - expert_heatmap)
        #_, Q_margin = model.get_dqn_actions(margin)
        #margin_loss = Q_margin - Q_expert # Nx4
        #margin_loss = model.margin_weight*torch.mean(margin_loss, axis=1, keepdim=False) # Nx1

        meta = {
                'Q': Q, 'nQ': nQ, 'hparams': model.hparams,
                'td_loss': td_loss, 
                'margin_loss': margin_loss,
                }

        visualize(batch, logits, tmap, nlogits, nhmap, naction, meta)

        break

        #if 1 in done:
        #    cv2.waitKey(0)
        #break
