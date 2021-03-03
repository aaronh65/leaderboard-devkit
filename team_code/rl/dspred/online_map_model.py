import sys, traceback
from datetime import datetime

import uuid
import argparse
import pathlib

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision
import numpy as np
import wandb

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from PIL import Image, ImageDraw
import cv2

from lbc.carla_project.src.models import SegmentationModel, RawController, SpatialSoftmax
from lbc.carla_project.src.utils.heatmap import ToHeatmap
from lbc.carla_project.src.common import COLOR

from rl.dspred.online_dataset import get_dataset

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
        x, y = action[i].cpu().numpy().astype(np.uint8)
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
        x, y = naction[i].cpu().numpy().astype(np.uint8)
        _ndraw.ellipse((x-2, y-2, x+2, y+2), (0,255,0))
        _ndraw.text((5, 10), f'action = ({x},{y})', textcolor)
        _ndraw.text((5, 20), f'nQ = {nQ[i].item():.2f}', textcolor)
        _ndraw.text((5, 30), f'done = {bool(done[i])}', textcolor)
        #_ntopdown = cv2.cvtColor(np.array(_ntopdown), cv2.COLOR_BGR2RGB)

        _combined = np.hstack((np.array(_topdown), np.array(_ntopdown)))
        #cv2.imshow(f'topdown{i}', cv2.cvtColor(_combined, cv2.COLOR_BGR2RGB))
        _combined = _combined.transpose(2,0,1)
        images.append((batch_loss[i].item(), torch.ByteTensor(_combined)))
    #cv2.waitKey(0)
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

        # RL stuff
        self.env = None
        self.n = 0 if not hasattr(hparams, 'n') else hparams.n
        self.discount = 0.99 ** (self.n + 1)
        self.criterion = torch.nn.MSELoss(reduction='none') # weights? prioritized replay?

        self.populate()

    # burn in
    def populate(self, steps=1000):
        pass

    
    def forward(self, topdown, target, debug=False):
        target_heatmap = self.to_heatmap(target, topdown)[:, None]
        out = self.net(torch.cat((topdown, target_heatmap), 1), heatmap=debug)

        if not debug:
            return out

        points, logits = out

        # extract action?

        return points, (logits, target_heatmap)

    def get_action(self, vmap):
        aim_vmap = torch.mean(vmap[:,0:2,:,:], dim=1) #(N, H, W)
        aim_vmap_flat = aim_vmap.view(aim_vmap.shape[:-2] + (-1,)) # (N, H*W)
        Q, aim_flat = torch.max(aim_vmap_flat, -1, keepdim=True)
        action = torch.cat((aim_flat % 256, aim_flat // 256), axis=1)
        return action, Q

    def training_step(self, batch, batch_nb):
        state, action, reward, next_state, done = batch
        
        # get currennt state's Q value
        topdown, target = state
        points, (vmap, hmap) = self.forward(topdown, target, debug=True)
        x_aim, y_aim = action[:,0], action[:,1] 
        aim_flat = torch.unsqueeze(y_aim * 256 + x_aim, dim=1) # (N, 1)
        aim_vmap = torch.mean(vmap[:,0:2,:,:], dim=1) # (N, H, W)
        aim_vmap_flat = aim_vmap.view(aim_vmap.shape[:-2] + (-1,)) # (N, H*W)
        Q = aim_vmap_flat.gather(1, aim_flat.long()) # (N, 1)

        # get next state's Q value
        ntopdown, ntarget = next_state
        with torch.no_grad():
            npoints, (nvmap, nhmap) = self.forward(ntopdown, ntarget, debug=True)
        naction, nQ = self.get_action(nvmap)

        target = reward + self.discount * nQ
        batch_loss = self.criterion(Q, target) # TD(n) error
        loss = batch_loss.mean()

        metrics = {f'TD({self.n}) loss': loss.item()}

        if batch_nb % 10 == 0:
            meta = {'Q': Q, 'nQ': nQ, 'batch_loss': batch_loss, 'hparams': self.hparams, 'n':self.n}
            images = visualize(batch, points, vmap, hmap, npoints, nvmap, nhmap, naction, meta)
            metrics['train_image'] = images

        self.logger.log_metrics(metrics, self.global_step)

        return {'loss': loss}

    # eval and no grad already set
    def validation_step(self, batch, batch_nb):
        state, action, reward, next_state, done = batch

        topdown, target = state
        points, (vmap, hmap) = self.forward(topdown, target, debug=True)
        ntopdown, ntarget = next_state
        npoints, (nvmap, nhmap) = self.forward(ntopdown, ntarget, debug=True)

        # get action in 1D form
        x_aim, y_aim = action[:,0], action[:,1] 
        aim_flat = torch.unsqueeze(y_aim * 256 + x_aim, dim=1)

        # average t=0.5, t=1.0: (N, 4, H, W) -> (N, H, W)
        aim_vmap = torch.mean(vmap[:,0:2,:,:], dim=1) # (N, H, W)
        aim_vmap_flat = aim_vmap.view(aim_vmap.shape[:-2] + (-1,)) # (N, H*W)
        Q = aim_vmap_flat.gather(1, aim_flat.long()) # (N, 1)
        
        # retrieve next action using next state's Q values
        naim_vmap = torch.mean(nvmap[:,0:2,:,:], dim=1) 
        naim_vmap_flat = naim_vmap.view(naim_vmap.shape[:-2] + (-1,))
        nQ, naim_flat = torch.max(naim_vmap_flat, -1, keepdim=True)
        naction = torch.cat((naim_flat % 256, naim_flat // 256), axis=1)

        target = reward + self.discount * nQ
        batch_val_loss = self.criterion(Q, target) # TD(n) error
        val_loss = batch_val_loss.mean()

        if batch_nb == 0:
            meta = {'Q': Q, 'nQ': nQ, 'batch_loss': batch_val_loss, 'hparams': self.hparams, 'n':self.n}
            images = visualize(batch, points, vmap, hmap, npoints, nvmap, nhmap, naction, meta)
            self.logger.log_metrics({'val_image': images}, self.global_step)
        
        return {'val_loss': val_loss.item()}
        
    def validation_epoch_end(self, batch_metrics):
        results = dict()

        for metrics in batch_metrics:
            for key in metrics:
                if key not in results:
                    results[key] = list()

                results[key].append(metrics[key])
        summary = {key: np.mean(val) for key, val in results.items()}
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
        print('train dataloader')
        return get_dataset(self.hparams.dataset_dir, True, self.hparams.batch_size, sample_by=self.hparams.sample_by, n=self.n)

    def val_dataloader(self):
        print('val dataloader')
        return get_dataset(self.hparams.dataset_dir, False, self.hparams.batch_size, sample_by=self.hparams.sample_by, n=self.n)


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

    trainer.fit(model)

    wandb.save(str(hparams.save_dir / '*.ckpt'))


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_epochs', type=int, default=40)
    parser.add_argument('--save_dir', type=pathlib.Path, default='checkpoints')
    parser.add_argument('--id', type=str, default=uuid.uuid4().hex) # replace with datetime

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
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    parsed.id = date_str
    save_dir = parsed.save_dir / 'debug' / parsed.id if parsed.debug else parsed.save_dir / parsed.id
    parsed.save_dir = save_dir
    parsed.save_dir.mkdir(parents=True, exist_ok=True)

    main(parsed)
