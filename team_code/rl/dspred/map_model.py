import os, yaml, copy
import argparse
import pathlib
from datetime import datetime
from tqdm import tqdm

import cv2
import wandb
import numpy as np
import torch, torchvision
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw

from lbc.carla_project.src.models import SegmentationModel, RawController, SpatialSoftmax
from lbc.carla_project.src.utils.heatmap import ToHeatmap, ToTemporalHeatmap
from lbc.carla_project.src.common import COLOR
from rl.dspred.offline_dataset import get_dataloader
#from rl.dspred.online_dataset import get_dataloader
 
HAS_DISPLAY = int(os.environ['HAS_DISPLAY'])
PROJECT_ROOT = os.environ['PROJECT_ROOT']
RESUME = f'{PROJECT_ROOT}/team_code/rl/config/weights/lbc_expert.ckpt'

# takes (N,3,H,W) topdown and (N,4,H,W) vmaps
# averages t=0.5s,1.0s vmaps and overlays it on topdown
@torch.no_grad()
def fuse_vmaps(topdown, vmap, temperature=10, alpha=0.75):

    vmap_mean = torch.mean(vmap[:,0:2,:,:], dim=1, keepdim=True) # N,1,H,W
    vmap_flat = vmap_mean.view(vmap_mean.shape[:-2] + (-1,)) # N,1,H*W
    vmap_prob = F.softmax(vmap_flat/temperature, dim=-1) # to prob
    vmap_norm = vmap_prob / torch.max(vmap_prob, dim=-1, keepdim=True)[0] # to [0,1]
    vmap_show = (vmap_norm * 256).view_as(vmap_mean).cpu().numpy().astype(np.uint8) # N,1,H,W
    vmap_show = np.repeat(vmap_show, repeats=3, axis=1).transpose((0,2,3,1)) # N,H,W,3
    fused = np.array(COLOR[topdown.argmax(1).cpu()]).astype(np.uint8) # N,H,W,3
    for i in range(fused.shape[0]):
        cv2.addWeighted(vmap_show[i], alpha, fused[i], 1, 0, fused[i])
    fused = fused.astype(np.uint8) # (N,H,W,3)
    return fused

# visualize each timestep's heatmap?
@torch.no_grad()
def visualize(batch, vmap, hmap, nvmap, nhmap, naction, meta, r=2):

    textcolor = (255,255,255)
    dqn_color = (65,105,225) # dark blue
    aim_color = (60,179,113) # dark green
    lbc_color = (178,34,34) # dark red
    #route_colors = [(0,255,0), (255,255,255), (255,0,0), (255,0,0)] 

    state, action, reward, next_state, done, info = batch
    hparams, td_loss, margin_loss = meta['hparams'], meta['td_loss'], meta['margin_loss']
    Q, nQ = meta['Q'], meta['nQ']
    discount = info['discount'].cpu()
    n = hparams.n

    topdown, target = state
    fused = fuse_vmaps(topdown, vmap, hparams.temperature, 1.0)

    ntopdown, ntarget = next_state
    nfused = fuse_vmaps(ntopdown, nvmap, hparams.temperature, 1.0)

    images = list()
    for i in range(min(action.shape[0], 32)):

        # current state
        _topdown = fused[i]
        _topdown[hmap[i][0].cpu() > 0.1] = 255
        _topdown = Image.fromarray(_topdown)
        _draw = ImageDraw.Draw(_topdown)
        _action = action[i].cpu().numpy().astype(np.uint8) # (4,2)
        if 'points_lbc' in info.keys():
            points_lbc = info['points_lbc']
            for x, y in points_lbc[i]:
                _draw.ellipse((x-r, y-r, x+r, y+r), lbc_color)
        if 'points_dqn' in info.keys():
            points_dqn = info['points_dqn']
            for x, y in points_dqn[i][:2]:
                _draw.ellipse((x-r, y-r, x+r, y+r), dqn_color)
        x, y = np.mean(_action[0:2], axis=0)
        _draw.ellipse((x-r, y-r, x+r, y+r), aim_color)

        _draw.text((5, 10), f'action = ({x},{y})', textcolor)
        _draw.text((5, 20), f'Q = {Q[i].item():.2f}', textcolor)
        _draw.text((5, 30), f'reward = {reward[i].item():.3f}', textcolor)
        _draw.text((5, 40), f'done = {bool(done[i])}', textcolor)

        # next state
        _ntopdown = nfused[i]
        _ntopdown[nhmap[i][0].cpu() > 0.1] = 255
        _ntopdown = Image.fromarray(_ntopdown)
        _ndraw = ImageDraw.Draw(_ntopdown)
        _naction = naction[i].cpu().numpy().astype(np.uint8) # (4,2)
        for x, y in _naction[0:2]:
            _ndraw.ellipse((x-r, y-r, x+r, y+r), dqn_color)
        x, y = np.mean(_naction[0:2], axis=0)
        _ndraw.ellipse((x-r, y-r, x+r, y+r), aim_color)

        _ndraw.text((5, 10), f'action = ({x},{y})', textcolor)
        _ndraw.text((5, 20), f'nQ = {nQ[i].item():.2f}', textcolor)
        _ndraw.text((5, 30), f'discount = {discount[i].item():.2f}', textcolor)
        _ndraw.text((5, 40), f'td_loss = {td_loss[i].item():.2f}', textcolor)
        _ndraw.text((5, 50), f'margin_loss = {margin_loss[i].item():.2f}', textcolor)

        _combined = np.hstack((np.array(_topdown), np.array(_ntopdown)))
        if HAS_DISPLAY:
            cv2.imshow(f'topdown{i}', cv2.cvtColor(_combined, cv2.COLOR_BGR2RGB))
        _combined = _combined.transpose(2,0,1)
        images.append((td_loss[i].item() + margin_loss[i].item(), torch.ByteTensor(_combined)))
        #images.append((td_loss[i].item(), torch.ByteTensor(_combined)))

    if HAS_DISPLAY:
        cv2.waitKey(1)

    images.sort(key=lambda x: x[0], reverse=True)
    result = torchvision.utils.make_grid([x[1] for x in images], nrow=4)
    result = wandb.Image(result.numpy().transpose(1,2,0))

    return images, result


# just needs to know if it's rolling out or nah
class MapModel(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams

        # model stuff
        self.to_heatmap = ToHeatmap(hparams.heatmap_radius)
        self.expert_heatmap = ToTemporalHeatmap(56)
        self.net = SegmentationModel(10, 4, hack=hparams.hack, temperature=hparams.temperature)
        self.controller = RawController(4)
        #self.criterion = torch.nn.MSELoss(reduction='none') # weights? prioritized replay?
        self.td_criterion = torch.nn.MSELoss(reduction='none') # weights? prioritized replay?
        self.margin_criterion = torch.nn.MSELoss(reduction='none')
        self.margin_weight = 100

        
    def setup_train(self, env, config):
        self.config = config
        self.env = env
        self.n = config.agent.n
        self.discount = 0.99 ** (self.n + 1)

        print('populating...')
        self.populate(config.agent.burn_timesteps)
        self.env.reset()

    # burn in
    def populate(self, steps):
        # make sure agent is burning in instead of inferencing
        self.env.hero_agent.burn_in = True
        done = False
        for step in tqdm(range(steps)):
            if done or step % 200 == 0:
                if step != 0:
                    self.env.cleanup()
                self.env.reset()
            reward, done = self.env.step()
        self.env.cleanup()
        self.env.hero_agent.burn_in = False

    # move to environment class
    def rollout(self, num_episodes=-1, num_steps=-1):
        assert num_episodes != -1 or num_steps != -1, \
                'specify either num steps or num epsiodes' 

        num_steps, num_episodes = 0, 0

        ## step environment
        self.eval()
        self.env.reset()
        self.env.hero_agent.burn_in = np.random.random() < self.config.agent.epsilon
        with torch.no_grad():
            for rollout_step in range(self.config.agent.rollout_steps):
                _reward, _done = self.env.step() # reward, done
                if _done:
                    self.env.cleanup()
                    self.env.reset()
        self.train()

    def forward(self, topdown, target, debug=True):
        target_heatmap = self.to_heatmap(target, topdown)[:, None]
        out = self.net(torch.cat((topdown, target_heatmap), 1), heatmap=debug)

        if not debug:
            return out

        points, logits = out

        # extract action?

        return (points, logits, target_heatmap)

    # two modes: sample, argmax?
    def get_dqn_actions(self, vmap, explore=False):
        #aim_vmap = torch.mean(vmap[:,0:2,:,:], dim=1) #(N, H, W)
        vmap_flat = vmap.view(vmap.shape[:-2] + (-1,)) # (N, 4, H*W)
        Q_all, action_flat = torch.max(vmap_flat, -1, keepdim=True) # (N, 4, 1)

        if explore:
            pass
            # figure out vectorized sampling...
            #probs = vmap_flat / Q_all
            #print(probs.shape)
            #action_flat = np.random.choice(
            #        np.arange(256*256), 
            #        size=(len(vmap_flat), 4, 1), 
            #        p=probs.cpu().numpy())
            #print(action_flat.shape)

        action = torch.cat((
            action_flat % 256,
            action_flat // 256),
            axis=2) # (N, C, 2)
        
        return action, Q_all # (N,4,2), (N,4,1)

    def get_Q_values(self, vmap, action):
        x, y = action[...,0], action[...,1] # Nx4
        action_flat = torch.unsqueeze(y*256 + x, dim=2) # (N,4, 1)
        vmap_flat = vmap.view(vmap.shape[:-2] + (-1,)) # (N, 4, H*W)
        Q_all = vmap_flat.gather(2, action_flat.long()) # (N, 4, 1)
        return Q_all # (N,4,1)

    def training_step(self, batch, batch_nb):
        metrics ={}

        state, action, reward, next_state, done, info = batch
        topdown, target = state
        points, vmap, hmap = self.forward(topdown, target, debug=True)
        Q_all = self.get_Q_values(vmap, action)
        Q = torch.mean(Q_all[:, :2], axis=1, keepdim=False)

        ntopdown, ntarget = next_state
        with torch.no_grad():
            npoints, nvmap, nhmap = self.forward(ntopdown, ntarget, debug=True)
        naction, nQ_all = self.get_dqn_actions(nvmap)
        nQ = torch.mean(nQ_all[:, :2], axis=1, keepdim=False)

        # td loss
        discount = info['discount']
        td_target = reward + discount * nQ * (1-done)
        td_loss = self.td_criterion(Q, td_target) # TD(n) error Nx1

        # expert margin loss
        expert_heatmap = self.expert_heatmap(info['points_lbc'], vmap)
        margin = vmap + (1 - expert_heatmap)
        _, Q_margin = self.get_dqn_actions(margin)
        Q_expert = self.get_Q_values(vmap, info['points_lbc'])
        margin_loss = Q_margin - Q_expert # Nx4
        margin_loss = self.margin_weight*torch.mean(margin_loss, axis=1, keepdim=False) # Nx1

        batch_loss = 0
        if not self.hparams.no_margin:
            batch_loss += margin_loss
        if not self.hparams.no_td:
            batch_loss += td_loss
        loss = torch.mean(batch_loss, dim=0)
        

        if batch_nb % 50 == 0:
            # TODO: handle debug images
            #if self.config.save_debug:
            #    img = cv2.cvtColor(np.array(self.env.hero_agent.debug_img), cv2.COLOR_RGB2BGR)
            #    metrics['debug_image'] = wandb.Image(img)

            meta = {
                    'Q': Q, 'nQ': nQ, 'hparams': self.hparams,
                    'td_loss': td_loss, 
                    'margin_loss': margin_loss,
                    }
            images, result = visualize(batch, vmap, hmap, nvmap, nhmap, naction, meta)
            metrics['train_image'] = result
            
        if self.logger != None:
            loss_metrics = {
                f'TD({self.hparams.n}) loss': td_loss.mean().item(),
                'margin_loss': margin_loss.mean().item(),
                'batch_loss': batch_loss.mean().item(),
                }

            self.logger.log_metrics(metrics, self.global_step)
            self.logger.log_metrics(loss_metrics, self.global_step)

        return {'loss': loss}

    # make this a validation episode rollout?
    def validation_step(self, batch, batch_nb):
        metrics ={}

        state, action, reward, next_state, done, info = batch
        topdown, target = state
        points, vmap, hmap = self.forward(topdown, target, debug=True)
        Q_all = self.get_Q_values(vmap, action)
        Q = torch.mean(Q_all[:, :2], axis=1, keepdim=False)

        ntopdown, ntarget = next_state
        npoints, nvmap, nhmap = self.forward(ntopdown, ntarget, debug=True)
        naction, nQ_all = self.get_dqn_actions(nvmap)
        nQ = torch.mean(nQ_all[:, :2], axis=1, keepdim=False)

        # td loss
        discount = info['discount']
        td_target = reward + discount * nQ * (1-done)
        td_loss = self.td_criterion(Q, td_target) # TD(n) error Nx1

        # expert margin loss
        expert_heatmap = self.expert_heatmap(info['points_lbc'], vmap)
        margin = vmap + (1 - expert_heatmap)
        _, Q_margin = self.get_dqn_actions(margin)
        Q_expert = self.get_Q_values(vmap, info['points_lbc'])
        margin_loss = Q_margin - Q_expert # Nx4
        margin_loss = self.margin_weight*torch.mean(margin_loss, axis=1, keepdim=False) # Nx1

        batch_loss = 0
        if not self.hparams.no_margin:
            batch_loss += margin_loss
        if not self.hparams.no_td:
            batch_loss += td_loss
        val_loss = torch.mean(batch_loss, axis=0)

                
        if self.logger != None:
            loss_metrics = {
                    f'TD({self.hparams.n}) loss': td_loss.mean().item(),
                    'margin_loss': margin_loss.mean().item(),
                    'batch_loss': batch_loss.mean().item(),
                    }

            meta = {
                    'Q': Q, 'nQ': nQ, 'hparams': self.hparams, 
                    'td_loss': td_loss,
                    'margin_loss': margin_loss,
                    }
            images, result = visualize(batch, vmap, hmap, nvmap, nhmap, naction, meta)
            metrics['val_image'] = result

            self.logger.log_metrics(metrics, self.global_step)
            self.logger.log_metrics(loss_metrics, self.global_step)
        return {'val_loss': val_loss}

    def validation_epoch_end(self, batch_metrics):
        results = dict()

        for metrics in batch_metrics:
            for key in metrics:
                if key not in results:
                    results[key] = list()
                results[key].append(metrics[key].mean().item())

        summary = {key: np.mean(val) for key, val in results.items()}
        if self.logger != None:
            self.logger.log_metrics(summary, self.global_step)

        return summary

    def configure_optimizers(self):
        optim = torch.optim.Adam(
                list(self.net.parameters()) + list(self.controller.parameters()),
                lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optim, mode='min', factor=0.5, patience=5, min_lr=1e-6,
                verbose=True)
        return [optim], [scheduler]

    def train_dataloader(self):
        return get_dataloader(self.hparams, is_train=True)

    # online val dataloaders spoof a length of N batches, and do N episode rollouts
    def val_dataloader(self):
        return get_dataloader(self.hparams, is_train=False)


# offline training

def main(args):

    logger = False
    if args.log:
        logger = WandbLogger(id=args.id, save_dir=str(args.save_dir), project='dqn_offline')
    checkpoint_callback = ModelCheckpoint(args.save_dir, save_top_k=1) # figure out what's up with this

    # resume and add a couple arguments
    model = MapModel.load_from_checkpoint(RESUME)
    model.hparams.max_epochs = args.max_epochs
    model.hparams.dataset_dir = args.dataset_dir
    model.hparams.batch_size = args.batch_size
    model.hparams.save_dir = args.save_dir
    model.hparams.n = args.n
    model.hparams.gamma = args.gamma
    model.hparams.num_workers = args.num_workers
    model.hparams.no_margin = args.no_margin
    model.hparams.no_td = args.no_td
    model.hparams.data_mode = 'offline'

    with open(args.save_dir / 'config.yml', 'w') as f:
        hparams_copy = copy.copy(vars(model.hparams))
        hparams_copy['save_dir'] = str(model.hparams.save_dir)
        del hparams_copy['id']
        yaml.dump(hparams_copy, f, default_flow_style=False, sort_keys=False)

    # offline trainer can use all gpus
    # when resuming, the network starts at epoch 36
    trainer = pl.Trainer(
        gpus=args.gpus, max_epochs=args.max_epochs,
        resume_from_checkpoint=RESUME,
        logger=logger,
        checkpoint_callback=checkpoint_callback,
        distributed_backend='dp',)

    trainer.fit(model)

    if args.log:
        wandb.save(str(args.save_dir / '*.ckpt'))



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('-D', '--debug', action='store_true')

    # Trainer args
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('-G', '--gpus', type=int, default=-1)
    
    parser.add_argument('--save_dir', type=pathlib.Path)
    parser.add_argument('--data_root', type=pathlib.Path, default='/data')
    parser.add_argument('--id', type=str, default=datetime.now().strftime("%Y%m%d_%H%M%S")) 
    #parser.add_argument('--offline', action='store_true', default=False)

    # Model args
    parser.add_argument('--heatmap_radius', type=int, default=5)
    #parser.add_argument('--command_coefficient', type=float, default=0.1)
    parser.add_argument('--temperature', type=float, default=10.0)
    parser.add_argument('--hack', action='store_true', default=False) # what is this again?
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--n', type=int, default=20)
    parser.add_argument('--sample_by', type=str, 
            choices=['none', 'even', 'speed', 'steer'], default='none')
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--no_margin', action='store_true')
    parser.add_argument('--no_td', action='store_true')

    # Program args
    parser.add_argument('--dataset_dir', type=pathlib.Path)
    parser.add_argument('--log', action='store_true')

    
    args = parser.parse_args()
    assert not (args.no_margin and args.no_td), 'no loss provided for training'

    if args.dataset_dir is None: # local
        #args.dataset_dir = '/data/leaderboard/data/rl/dspred/debug/20210311_143718'
        args.dataset_dir = '/data/leaderboard/data/rl/dspred/20210311_213726'

    suffix = f'debug/{args.id}' if args.debug else args.id
    save_root = args.data_root / f'leaderboard/training/rl/dspred/{suffix}'

    args.save_dir = save_root
    args.save_dir.mkdir(parents=True, exist_ok=True)

    main(args)
