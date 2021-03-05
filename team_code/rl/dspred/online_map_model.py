import sys, traceback
from datetime import datetime
import pickle as pkl

import uuid
import argparse
import pathlib

import numpy as np
import cv2
import wandb
from PIL import Image, ImageDraw

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader


from lbc.carla_project.src.models import SegmentationModel, RawController, SpatialSoftmax
from lbc.carla_project.src.utils.heatmap import ToHeatmap
from lbc.carla_project.src.common import COLOR

from rl.dspred.online_dataset import get_dataloader
#from rl.dspred.online_dataset import CarlaDataset
 

# takes (N,3,H,W) topdown and (N,4,H,W) vmaps
# averages t=0.5s,1.0s vmaps and overlays it on topdown
@torch.no_grad()
def fuse_vmaps(topdown, vmap, hparams, alpha=0.75):

    vmap_mean = torch.mean(vmap[:,0:2,:,:], dim=1, keepdim=True) # N,1,H,W
    vmap_flat = vmap_mean.view(vmap_mean.shape[:-2] + (-1,)) # N,1,H*W
    vmap_prob = F.softmax(vmap_flat/hparams.temperature, dim=-1) # to prob
    vmap_norm = vmap_prob / torch.max(vmap_prob, dim=-1, keepdim=True)[0] # to [0,1]
    vmap_show = (vmap_norm * 256).view_as(vmap_mean).cpu().numpy().astype(np.uint8) # N,1,H,W
    vmap_show = np.repeat(vmap_show, repeats=3, axis=1).transpose((0,2,3,1)) # N,H,W,3
    fused = np.array(COLOR[topdown.argmax(1).cpu()]).astype(np.uint8) # N,H,W,3
    for i in range(fused.shape[0]):
        cv2.addWeighted(vmap_show[i], 0.75, fused[i], 1, 0, fused[i])
    fused = fused.astype(np.uint8)
    return fused

@torch.no_grad()
def visualize(batch, points, vmap, hmap, npoints, nvmap, nhmap, naction, meta):
    images = list()

    state, action, reward, next_state, done = batch
    topdown, target = state
    ntopdown, target = next_state
    hparams, batch_loss, n = meta['hparams'], meta['batch_loss'], meta['n']
    Q, nQ = meta['Q'], meta['nQ']

    fused = fuse_vmaps(topdown, vmap, hparams, 1.0)
    nfused = fuse_vmaps(ntopdown, nvmap, hparams, 1.0)
    points = (points + 1) / 2 * 256 # [-1, 1] -> [0, 256]
    npoints = (npoints + 1) / 2 * 256 # [-1, 1] -> [0, 256]

    textcolor = (255,255,255)
    for i in range(action.shape[0]):

        # current state
        _topdown = fused[i]
        _topdown[hmap[i][0].cpu() > 0.1] = 255
        _topdown = Image.fromarray(_topdown)
        _draw = ImageDraw.Draw(_topdown)
        for x, y in points[i].cpu().numpy():
            _draw.ellipse((x-2, y-2, x+2, y+2), (255,0,0))
        #x, y = action[i].cpu().numpy().astype(np.uint8)
        _draw.ellipse((x-2, y-2, x+2, y+2), (0,255,0))
        _draw.text((5, 10), f'action = ({x},{y})', textcolor)
        _draw.text((5, 20), f'Q = {Q[i].item():2f}', textcolor)
        _draw.text((5, 30), f'reward = {reward[i].item():.3f}', textcolor)
        _draw.text((5, 40), f'TD({n}) err = {batch_loss[i].item():.2f}', textcolor)
        #_topdown = cv2.cvtColor(np.array(_topdown), cv2.COLOR_BGR2RGB)

        # next state
        _ntopdown = nfused[i]
        _ntopdown[nhmap[i][0].cpu() > 0.1] = 255
        _ntopdown = Image.fromarray(_ntopdown)
        _ndraw = ImageDraw.Draw(_ntopdown)
        for x, y in npoints[i].cpu().numpy():
            _ndraw.ellipse((x-2, y-2, x+2, y+2), (255,0,0))
        #x, y = naction[i].cpu().numpy().astype(np.uint8)
        _ndraw.ellipse((x-2, y-2, x+2, y+2), (0,255,0))
        _ndraw.text((5, 10), f'action = ({x},{y})', textcolor)
        _ndraw.text((5, 20), f'nQ = {nQ[i].item():.2f}', textcolor)
        _ndraw.text((5, 30), f'done = {bool(done[i])}', textcolor)
        #_ntopdown = cv2.cvtColor(np.array(_ntopdown), cv2.COLOR_BGR2RGB)

        _combined = np.hstack((np.array(_topdown), np.array(_ntopdown)))
        #cv2.imshow(f'topdown{i}', cv2.cvtColor(_combined, cv2.COLOR_BGR2RGB))
        _combined = _combined.transpose(2,0,1)
        images.append((batch_loss[i].item(), torch.ByteTensor(_combined)))

    #cv2.waitKey(1)
    images.sort(key=lambda x: x[0], reverse=True)
    result = torchvision.utils.make_grid([x[1] for x in images], nrow=3)
    result = wandb.Image(result.numpy().transpose(1,2,0))

    return result



class MapModel(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams

        # model stuff
        self.to_heatmap = ToHeatmap(hparams.heatmap_radius)
        self.net = SegmentationModel(10, 4, hack=hparams.hack, temperature=hparams.temperature)
        self.controller = RawController(4)

        
    def setup_train(self, env, config):
        self.config = config
        self.env = env
        self.n = config.agent.n
        self.discount = 0.99 ** (self.n + 1)
        self.criterion = torch.nn.MSELoss(reduction='none') # weights? prioritized replay?

        self.populate(config.agent.burn_timesteps)
        print('done populating')

        #with open('buffer.pkl', 'wb') as f:
        #    pkl.dump(self.env.buffer, f)
        #with open('buffer.pkl', 'rb') as f:
        #    self.env.buffer = pkl.load(f)

        self.env.reset()
        self.last_loss = 0

    # burn in
    def populate(self, steps=100):
        # make sure agent is burning in instead of inferencing
        self.env.hero_agent.burn_in = True
        done = False
        for step in range(steps):
            if done or step % 200 == 0:
                if step != 0:
                    self.env.cleanup()
                self.env.reset()
            reward, done = self.env.step()
        self.env.cleanup()
        self.env.hero_agent.burn_in = False

    def forward(self, topdown, target, debug=False):
        target_heatmap = self.to_heatmap(target, topdown)[:, None]
        out = self.net(torch.cat((topdown, target_heatmap), 1), heatmap=debug)

        if not debug:
            return out

        points, logits = out

        # extract action?

        return points, (logits, target_heatmap)

    def get_actions(self, vmap):
        #aim_vmap = torch.mean(vmap[:,0:2,:,:], dim=1) #(N, H, W)
        vmap_flat = vmap.view(vmap.shape[:-2] + (-1,)) # (N, 4, H*W)
        Q_all, action_flat = torch.max(vmap_flat, -1, keepdim=True) # (N, 4, 1)
        action = torch.cat((
            action_flat % 256,
            action_flat // 256),
            axis=2) # (N, C, 2)
            #torch.remainder(action_flat, 256),  # aim_flat % 256 = x (column)
            #torch.floor_divide(action_flat, 256)),  # aim_flat // 256 = y (row)
        
        return action, Q_all.squeeze() # (N,4,2), (N,4)

    def get_action_values(self, vmap, action):
        # action is (N, C, 2)
        #action_xy = action.view((-1, 4, 2)) # Nx4x2
        x, y = action[...,0], action[...,1] # Nx4
        action_flat = torch.unsqueeze(y*256 + x, dim=2) # (N,4, 1)
        #action_flat = y*256 + x # Nx4
        vmap_flat = vmap.view(vmap.shape[:-2] + (-1,)) # (N, 4, H*W)
        Q_all = vmap_flat.gather(2, action_flat.long()) # (N, 4, 1)
        return Q_all.squeeze() # (N,4)

    def training_step(self, batch, batch_nb):

        ## train on batch
        state, action, reward, next_state, done = batch
        
        # get Q values
        topdown, target = state
        points, (vmap, hmap) = self.forward(topdown, target, debug=True)
        Q_all = self.get_action_values(vmap, action)

        ntopdown, ntarget = next_state
        with torch.no_grad():
            npoints, (nvmap, nhmap) = self.forward(ntopdown, ntarget, debug=True)
        naction, nQ_all = self.get_actions(nvmap)

        # choose t=1,2
        Q = torch.mean(Q_all[:, :2], axis=1, keepdim=True)
        nQ = torch.mean(nQ_all[:, :2], axis=1, keepdim=True)

        # compute loss and metrics
        target = reward + self.discount * nQ
        batch_loss = self.criterion(Q, target) # TD(n) error
        loss = batch_loss.mean()

        metrics = {f'TD({self.n}) loss': loss.item()}
        if batch_nb % 10 == 0:
            meta = {'Q': Q, 'nQ': nQ, 'batch_loss': batch_loss, 'hparams': self.hparams, 'n':self.n}
            images = visualize(batch, points, vmap, hmap, npoints, nvmap, nhmap, naction, meta)
            metrics['train_image'] = images
        self.last_loss = loss
        if self.logger != None:
            self.logger.log_metrics(metrics, self.global_step)

        ## step environment
        _reward, _done = self.env.step() # reward, done
        if _done:
            self.env.cleanup()
            self.env.reset()

        return {'loss': loss}

    def validation_step(self):
        return self.last_loss.item()

    def configure_optimizers(self):
        optim = torch.optim.Adam(
                list(self.net.parameters()) + list(self.controller.parameters()),
                lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optim, mode='min', factor=0.5, patience=5, min_lr=1e-6,
                verbose=True)
        return [optim], [scheduler]

    def train_dataloader(self):
        return get_dataloader(self.env.buffer, self.config.agent.batch_size)



def main(hparams):

    logger = False
    if hparams.log:
        logger = WandbLogger(id=hparams.id, save_dir=str(hparams.save_dir), project='dqn')
    checkpoint_callback = ModelCheckpoint(hparams.save_dir, save_top_k=1)

    # pass env into map model constructor
    try:
        resume = '/home/aaron/workspace/carla/leaderboard-devkit/team_code/rl/config/weights/map_model.ckpt'
        model = MapModel.load_from_checkpoint(resume)
        model.hparams.dataset_dir = hparams.dataset_dir
        model.hparams.batch_size = hparams.batch_size
        model.n = hparams.n
        print(f'resuming from {resume}')

    except Exception as e:
        traceback.print_exception(type(e), e, e.__traceback__)
        resume = None

    trainer = pl.Trainer(
            gpus=-1, max_epochs=hparams.max_epochs,
            resume_from_checkpoint=resume,
            logger=logger,
            checkpoint_callback=checkpoint_callback)
            #check_val_every

    trainer.fit(model)

    wandb.save(str(hparams.save_dir / '*.ckpt'))


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_epochs', type=int, default=40)
    parser.add_argument('--save_dir', type=pathlib.Path, default='checkpoints')
    parser.add_argument('--id', type=str, default=datetime.now().strftime("%Y%m%d_%H%M%S")) # replace with datetime

    parser.add_argument('--heatmap_radius', type=int, default=5)
    parser.add_argument('--sample_by', type=str, choices=['none', 'even', 'speed', 'steer'], default='even')
    parser.add_argument('--command_coefficient', type=float, default=0.1)
    parser.add_argument('--temperature', type=float, default=10.0)
    parser.add_argument('--hack', action='store_true', default=False)

    # Data args.
    parser.add_argument('--dataset_dir', type=pathlib.Path, required=True)
    parser.add_argument('--batch_size', type=int, default=16)

    # Optimizer args.
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.0)

    # RL args
    #parser.add_argument('--offline', action='store_true', default=False)
    parser.add_argument('--n', type=int, default=0)
    parser.add_argument('--log', action='store_true')
    parser.add_argument('-d', '--debug', action='store_true')

    parsed = parser.parse_args()
    save_dir = parsed.save_dir / 'debug' / parsed.id if parsed.debug else parsed.save_dir / parsed.id
    parsed.save_dir = save_dir
    parsed.save_dir.mkdir(parents=True, exist_ok=True)

    main(parsed)
