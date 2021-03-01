import sys

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

from rl.dspred.dataset import get_dataset

@torch.no_grad()
def visualize(topdown, hmap, vmap, action, ntopdown, nhmap, nvmap, naction, reward, points, npoints):
    images = list()
    
    # first process value maps and extract next state aim
    vmap_flat = vmap.view(vmap.shape[:-2] + (-1,))
    vmap_prob = F.softmax(vmap_flat/10, dim=-1)
    weights = vmap_prob.view_as(vmap)
    x = (weights.sum(-2)*torch.linspace(-1, 1, vmap.shape[-1]).type_as(vmap)).sum(-1)
    y = (weights.sum(-1)*torch.linspace(-1, 1, vmap.shape[-2]).type_as(vmap)).sum(-1)
    x = np.clip((x.cpu().numpy() + 1) / 2 * 256, 0, 256)
    y = np.clip((y.cpu().numpy() + 1) / 2 * 256, 0, 256)
    vmap_show = vmap_prob / torch.max(vmap_prob, dim=-1, keepdim=True)[0]
    vmap_show = vmap_show.view_as(vmap)
    vmap_mean = torch.mean(vmap_show[:,0:2,:,:], dim=1) 

    # next state
    nvmap_flat = nvmap.view(nvmap.shape[:-2] + (-1,))
    nvmap_prob = F.softmax(nvmap_flat/10, dim=-1)
    nweights = nvmap_prob.view_as(nvmap)
    nx = (nweights.sum(-2)*torch.linspace(-1, 1, nvmap.shape[-1]).type_as(nvmap)).sum(-1)
    ny = (nweights.sum(-1)*torch.linspace(-1, 1, nvmap.shape[-2]).type_as(nvmap)).sum(-1)
    nx = np.clip((nx.cpu().numpy() + 1) / 2 * 256, 0, 256)
    ny = np.clip((ny.cpu().numpy() + 1) / 2 * 256, 0, 256)
    nvmap_show = nvmap_prob / torch.max(nvmap_prob, dim=-1, keepdim=True)[0]
    nvmap_show = nvmap_show.view_as(nvmap)
    nvmap_mean = torch.mean(nvmap_show[:,0:2,:,:], dim=1) 

    #points = np.clip((points.cpu().numpy() + 1) / 2 * 256, 0, 256)
    #npoints = np.clip((npoints.cpu().numpy() + 1) / 2 * 256, 0, 256)

    for i in range(topdown.shape[0]): # batch size N

        _topdown, _hmap, _vmap, _action = topdown[i], hmap[i], vmap_show[i], action[i]
        _topdown = COLOR[_topdown.argmax(0).cpu().numpy()]
        _topdown[_hmap.cpu().numpy().squeeze() > 0.1] = 255
        xaim, yaim = action[i,0], action[i,1]
        vmap_stack = []
        for t in range(len(_vmap)):
            vmap_out = _topdown.copy()
            vmap_view = _vmap[t].cpu().numpy() * 256
            vmap_view = np.dstack((vmap_view, vmap_view, vmap_view))
            vmap_view = vmap_view.astype(np.uint8)
            cv2.addWeighted(vmap_view, 0.75, vmap_out, 1, 0, vmap_out)
            vmap_im = Image.fromarray(vmap_out)
            vmap_imdraw = ImageDraw.Draw(vmap_im)
            _x, _y = x[i,t], y[i,t]
            vmap_imdraw.ellipse((xaim-2,yaim-2,xaim+2,yaim+2), (0,191,255))
            vmap_imdraw.ellipse((_x-2,_y-2,_x+2,_y+2), (255,0,0))
            vmap_im = cv2.cvtColor(np.array(vmap_im), cv2.COLOR_BGR2RGB)
            vmap_stack.append(vmap_im)
        vmap_comb = np.hstack(vmap_stack)
        cv2.imshow('vmaps', vmap_comb)


        _topdown = Image.fromarray(_topdown)
        _topdown_draw = ImageDraw.Draw(_topdown)
        #_x, _y = np.mean(x[i,0:2]), np.mean(y[i,0:2])
        #for _x, _y in zip(x[i], y[i]):
        #    _topdown_draw.ellipse((_x-2, _y-2, _x+2, _y+2), (0,255,0))
        _topdown_draw.text((5, 10), f'reward = {reward[i].item():.5f}', (255,255,255))
        _topdown = cv2.cvtColor(np.array(_topdown), cv2.COLOR_BGR2RGB)

        # next state
        _ntopdown, _nhmap, _naction = ntopdown[i], nhmap[i], naction[i]
        _ntopdown = COLOR[_ntopdown.argmax(0).cpu().numpy()]
        _ntopdown[_hmap.cpu().numpy().squeeze() > 0.1] = 255
        _ntopdown = Image.fromarray(_ntopdown)

        _ntopdown_draw = ImageDraw.Draw(_ntopdown)
        #_nx, _ny = np.mean(nx[i,0:2]), np.mean(ny[i,0:2])
        #for _nx, _ny in zip(nx[i], ny[i]):
        #    _ntopdown_draw.ellipse((_nx-2, _ny-2, _nx+2, _ny+2), (0,255,0))

        _ntopdown = cv2.cvtColor(np.array(_ntopdown), cv2.COLOR_BGR2RGB)

        
        _combined = np.hstack((_topdown, _ntopdown))
        images.append(_combined)
        cv2.imshow('topdown', _combined)
        cv2.waitKey(0)

    images.sort(key=lambda x: x[0], reverse=True)

    result = torchvision.utils.make_grid([x[1] for x in images], nrow=4)
    result = wandb.Image(result.numpy().transpose(1, 2, 0))

    return result



class MapModel(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams

        self.to_heatmap = ToHeatmap(hparams.heatmap_radius)
        self.net = SegmentationModel(10, 4, hack=hparams.hack, temperature=hparams.temperature)
        self.controller = RawController(4)
        self.criterion = torch.nn.MSELoss()
        #self.discount = 0.99 ** (self.hparams.n + 1)
        self.discount = 0.99 ** (0 + 1)

    def forward(self, topdown, target, debug=False):
        target_heatmap = self.to_heatmap(target, topdown)[:, None]
        out = self.net(torch.cat((topdown, target_heatmap), 1), heatmap=debug)

        if not debug:
            return out

        points, logits = out
        return points, (logits, target_heatmap)

    def training_step(self, batch, batch_nb):
        state, action, reward, next_state, done = batch
        # retrieve Q prediction
        topdown, target = state
        points, (vmap, target_hmap) = self.forward(topdown, target, debug=True)
        #vmap = (vmap + 1) * 256
        aim_vmap = torch.mean(vmap[:,0:2,:,:], dim=1) 

        # get action in 1D form
        x_aim, y_aim = action[:,0], action[:,1]
        action_flat = torch.unsqueeze(y_aim * 256 + x_aim, dim=1)

        # flatten last two dims and index the action to get Q
        # average t=0.5, t=1.0: (N, 4, H, W) -> (N, H, W)
        aim_vmap_flat = aim_vmap.view(aim_vmap.shape[:-2] + (-1,)) # (N, W*H)
        Q = aim_vmap_flat.gather(1, action_flat.long())

        ntopdown, ntarget = next_state
        with torch.no_grad(): # no gradients on target right?
            npoints, (nvmap, _) = self.forward(ntopdown, ntarget, debug=True)
        #nvmap = (nvmap + 1) * 256
        naim_vmap = torch.mean(nvmap[:,0:2,:,:], dim=1) 
        naim_vmap_flat = naim_vmap.view(naim_vmap.shape[:-2] + (-1,))
        nQ, naction_flat = torch.max(naim_vmap_flat, -1, keepdim=True)
        naction = torch.cat((naction_flat % 256, naction_flat // 256), axis=1)

        target = reward + self.discount * nQ
        loss = self.criterion(Q, target) # TD(n) error

        return {'loss': loss}

    def validation_step(self, batch, batch_nb):
        self.eval()
        state, action, reward, next_state, done = batch

        # retrieve Q prediction
        topdown, target = state
        ntopdown, ntarget = next_state
        with torch.no_grad():
            points, (vmap, hmap) = self.forward(topdown, target, debug=True)
        points = (points + 1) / 2 * 256
        with torch.no_grad(): # no gradients on target right?
            npoints, (nvmap, nhmap) = self.forward(ntopdown, ntarget, debug=True)

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
            nheatmap = self.to_heatmap(ntarget[i:i+1], ntopdown[i:i+1]).squeeze().cpu()
            _ntopdown[nheatmap > 0.1] = 255
            _ntopdown = cv2.cvtColor(_ntopdown, cv2.COLOR_BGR2RGB)

            _combined = np.hstack((_topdown, _ntopdown))
            cv2.imshow(f'topdown {i}', _combined)
        cv2.waitKey(0)

#        if True:
#            _topdown = COLOR[topdown[0].argmax(0).cpu().numpy()]
#            heatmap = self.to_heatmap(target, topdown)[0].cpu().numpy()
#            _topdown[heatmap.squeeze() > 0.1] = 255
#            _topdown = Image.fromarray(_topdown)
#
#            _draw = ImageDraw.Draw(_topdown)
#            x, y = action[0].cpu().squeeze().numpy().astype(np.uint8)
#            _draw.ellipse((x-2, y-2, x+2, y+2), (0,191,255))
#            _draw.text((5, 10), f'reward = {reward[0].item():.5f}', (255,255,255))
#            for x, y in points_plot[0]:
#                _draw.ellipse((x-2, y-2, x+2, y+2), (255,0,0))
#            _topdown = cv2.cvtColor(np.array(_topdown), cv2.COLOR_BGR2RGB)
#
#            # next state
#            _ntopdown = COLOR[ntopdown[0].argmax(0).cpu().numpy()]
#            nheatmap = self.to_heatmap(ntarget, ntopdown)[0].cpu().numpy()
#            _ntopdown[nheatmap.squeeze() > 0.1] = 255
#            _ntopdown = cv2.cvtColor(_ntopdown, cv2.COLOR_BGR2RGB)
#
#            _combined = np.hstack((_topdown, _ntopdown))
#            cv2.imshow('topdown', _combined)
#            cv2.waitKey(0)
#

        # get action in 1D form
        #x_aim, y_aim = action[:,0], action[:,1]
        #action_flat = torch.unsqueeze(y_aim * 256 + x_aim, dim=1)

        ## average t=0.5, t=1.0: (N, 4, H, W) -> (N, H, W)
        #aim_vmap = torch.mean(vmap[:,0:2,:,:], dim=1) 
        #aim_vmap_flat = aim_vmap.view(aim_vmap.shape[:-2] + (-1,)) # (N, H*W)
        #Q = aim_vmap_flat.gather(1, action_flat.long())

        #ntopdown, ntarget = next_state
        #with torch.no_grad(): # no gradients on target right?
        #    npoints, (nvmap, ntgt_hmap) = self.forward(ntopdown, ntarget, debug=True)
        #naim_vmap = torch.mean(nvmap[:,0:2,:,:], dim=1) 
        #naim_vmap_flat = naim_vmap.view(naim_vmap.shape[:-2] + (-1,))
        #nQ, naction_flat = torch.max(naim_vmap_flat, -1, keepdim=True)
        #naction = torch.cat((naction_flat % 256, naction_flat // 256), axis=1)

        #target = reward + self.discount * nQ
        #val_loss = self.criterion(Q, target).mean() # TD(n) error

        ## WRITE VISUALIZATION
        #visualize(
        #        topdown, tgt_hmap, vmap, action, 
        #        ntopdown, ntgt_hmap, nvmap, naction, 
        #        reward, points, npoints)

        ## this state
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
        return get_dataset(self.hparams.dataset_dir, True, self.hparams.batch_size, sample_by=self.hparams.sample_by, n=self.hparams.n)

    def val_dataloader(self):
        print('val dataloader')
        return get_dataset(self.hparams.dataset_dir, False, self.hparams.batch_size, sample_by=self.hparams.sample_by, n=self.hparams.n)


def main(hparams):
    model = MapModel(hparams)

    logger = False
    if hparams.log:
        logger = WandbLogger(id=hparams.id, save_dir=str(hparams.save_dir), project='dqn')
    checkpoint_callback = ModelCheckpoint(hparams.save_dir, save_top_k=1)

    try:
        #resume_from_checkpoint = sorted(hparams.save_dir.glob('*.ckpt'))[-1]
        resume_from_checkpoint = '/home/aaron/workspace/carla/leaderboard-devkit/team_code/rl/config/weights/map_model.ckpt'
    except:
        resume_from_checkpoint = None
    trainer = pl.Trainer(
            gpus=-1, max_epochs=hparams.max_epochs,
            resume_from_checkpoint=resume_from_checkpoint,
            logger=logger,
            checkpoint_callback=checkpoint_callback)

    model.load_from_checkpoint(resume_from_checkpoint)
    trainer.fit(model)

    #wandb.save(str(hparams.save_dir / '*.ckpt'))


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--save_dir', type=pathlib.Path, default='checkpoints')
    parser.add_argument('--id', type=str, default=uuid.uuid4().hex) # replace with datetime

    parser.add_argument('--heatmap_radius', type=int, default=5)
    parser.add_argument('--sample_by', type=str, choices=['none', 'even', 'speed', 'steer'], default='even')
    parser.add_argument('--command_coefficient', type=float, default=0.1)
    parser.add_argument('--temperature', type=float, default=10.0)
    parser.add_argument('--hack', action='store_true', default=False)

    # Data args.
    parser.add_argument('--dataset_dir', type=pathlib.Path, required=True)
    parser.add_argument('--batch_size', type=int, default=5)

    # Optimizer args.
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.0)

    # RL args
    #parser.add_argument('--offline', action='store_true', default=False)
    parser.add_argument('--n', type=int, default=0)
    parser.add_argument('--log', action='store_true')

    parsed = parser.parse_args()
    parsed.save_dir = parsed.save_dir / parsed.id
    parsed.save_dir.mkdir(parents=True, exist_ok=True)

    main(parsed)
