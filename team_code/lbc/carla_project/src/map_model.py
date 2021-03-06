import argparse, copy, shutil, yaml
from pathlib import Path
from datetime import datetime

from PIL import Image, ImageDraw
import cv2, numpy as np

import torch, torchvision, wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from lbc.carla_project.src.models import SegmentationModel, RawController
from lbc.carla_project.src.utils.heatmap import ToHeatmap
from lbc.carla_project.src.dataset import get_dataset
from lbc.carla_project.src.common import COLOR, CONVERTER

from misc.utils import *

#text_color = (255,255,255)
#aim_color = (60,179,113) # dark green
#lbc_color = (178,34,34) # dark red
route_colors = [(255,255,255), (112,128,144), (47,79,79), (47,79,79)] 
HAS_DISPLAY=True

@torch.no_grad()
# N,C,H,W
def viz_weights(topdown, target, points, weights, loss_point=None, alpha=0.5, use_wandb=False):
    n,c,_ = points.shape
    points = points.clone().detach().cpu().numpy() # N,C,2
    points = (points + 1) / 2 * 256
    points = np.clip(points, 0, 256)
    target = target.clone().cpu().numpy()

    if loss_point is None:
        loss_point = np.zeros(n)

    # process topdown
    topdown = COLOR[topdown.argmax(1).clone().detach().cpu().numpy()] #N,H,W,3
    weights = spatial_norm(weights)*255
    weights = weights.cpu().numpy().astype(np.uint8) #N,4,H,W

    images = list()
    for i in range(min(n, 4)):
        _topdown = Image.fromarray(topdown[i]) # H,W,3
        img_stack = list()
        _tx, _ty = target[i]
        _ax, _ay = np.mean(points[i,0:2], axis=0)
        for _x, _y in points[i]:
            _td = _topdown.copy()
            _draw = ImageDraw.Draw(_td)
            _draw.ellipse((_x-2, _y-2, _x+2, _y+2), (255,0,0))
            _draw.ellipse((_ax-2, _ay-2, _ax+2, _ay+2), (0,255,0))
            _draw.ellipse((_tx-2, _ty-2, _tx+2, _ty+2), route_colors[1])
            img_stack.append(np.array(_td))
        _topdown_tiled = np.hstack(img_stack)
        _wgts = np.hstack([wgt for wgt in weights[i]]) #H,W*4
        _wgts = np.expand_dims(_wgts, 2) #H,W*4,1
        _wgts_tiled = np.tile(_wgts, (1,1,3))#H,W*4,3

        out = cv2.addWeighted(_wgts_tiled, alpha, _topdown_tiled, 1, 0)
        out = np.array(out)
        images.append((loss_point[i], out.transpose(2,0,1)))

    images.sort(key=lambda x: x[0], reverse=True)
    images = [x[1] for x in images]
    images = torchvision.utils.make_grid(
                [torch.ByteTensor(x) for x in images], nrow=1)
    images = images.numpy().transpose(1,2,0)
    if use_wandb:
        images = wandb.Image(images)
    else:
        images = cv2.cvtColor(images, cv2.COLOR_RGB2BGR)
        if HAS_DISPLAY:
            cv2.imshow('debug', images)
    return images

@torch.no_grad()
def viz_td(batch, out, between, out_cmd, loss_point, loss_cmd, use_wandb=False):
    images = list()

    for i in range(out.shape[0]):
        _loss_point = loss_point[i]
        _loss_cmd = loss_cmd[i]
        _out = out[i]
        _out_cmd = out_cmd[i]
        _between = between[i]

        #rgb, topdown, points, target, actions, meta = [x[i] for x in batch]
        rgb, topdown, points, target, actions, meta = [x[i] for x in batch]
        meta = meta.cpu().numpy().astype(int)
        meta = decode_str(meta)
        #meta = 'test'

        _rgb = np.uint8(rgb.detach().cpu().numpy().transpose(1, 2, 0) * 255)
        #_target_heatmap = np.uint8(target_heatmap[i].detach().squeeze().cpu().numpy() * 255)
        #_target_heatmap = np.stack(3 * [_target_heatmap], 2)
        #_target_heatmap = Image.fromarray(_target_heatmap)
        _topdown = Image.fromarray(COLOR[topdown.argmax(0).detach().cpu().numpy()])
        _draw = ImageDraw.Draw(_topdown)

        _draw.ellipse((target[0]-2, target[1]-2, target[0]+2, target[1]+2), route_colors[1])

        for x, y in points:
            x = (x + 1) / 2 * 256
            y = (y + 1) / 2 * 256

            _draw.ellipse((x-2, y-2, x+2, y+2), (0, 0, 255))

        for x, y in _out:
            x = (x + 1) / 2 * 256
            y = (y + 1) / 2 * 256

            _draw.ellipse((x-2, y-2, x+2, y+2), (255, 0, 0))

        #for x, y in _between:
        #    x = (x + 1) / 2 * 256
        #    y = (y + 1) / 2 * 256

        #    _draw.ellipse((x-1, y-1, x+1, y+1), (0, 255, 0))

        _draw.text((5, 10), 'Point: %.3f' % _loss_point)
        _draw.text((5, 30), 'Command: %.3f' % _loss_cmd)
        _draw.text((5, 50), 'Meta: %s' % meta)

        _draw.text((5, 90), 'Raw: %.3f %.3f' % tuple(actions))
        _draw.text((5, 110), 'Pred: %.3f %.3f' % tuple(_out_cmd))

        image = np.array(_topdown).transpose(2, 0, 1)
        images.append((_loss_cmd, torch.ByteTensor(image)))

    images.sort(key=lambda x: x[0], reverse=True)
    images = [x[1] for x in images]
    images = torchvision.utils.make_grid([torch.ByteTensor(x) for x in images], nrow=4)
    images = images.numpy().transpose(1, 2, 0)

    if use_wandb:
        images = wandb.Image(images)
    elif HAS_DISPLAY:
        cv2.imshow('debug', images)

    return images


class MapModel(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams

        self.to_heatmap = ToHeatmap(hparams.heatmap_radius)
        self.net = SegmentationModel(10, 4, batch_norm=True, hack=hparams.hack)
        self.controller = RawController(4)
        self.register_buffer('temperature', torch.Tensor([self.hparams.temperature]))

    def forward(self, topdown, target): # save global step?

        target_heatmap = self.to_heatmap(target, topdown)[:, None]
        input = torch.cat((topdown, target_heatmap), 1)
        points, logits = self.net(input, temperature=self.temperature.item())
        return points, logits, target_heatmap

    def training_step(self, batch, batch_nb):
        img, topdown, points, target, actions, meta = batch
        points_lbc, logits, target_heatmap= self.forward(topdown, target)

        alpha = torch.rand(points_lbc.shape).type_as(points_lbc)
        between = alpha * points_lbc + (1-alpha) * points
        out_cmd = self.controller(between)

        loss_point = torch.nn.functional.l1_loss(points_lbc, points, reduction='none').mean((1, 2))
        loss_cmd_raw = torch.nn.functional.l1_loss(out_cmd, actions, reduction='none')

        loss_cmd = loss_cmd_raw.mean(1)
        loss = (loss_point + self.hparams.command_coefficient * loss_cmd).mean()
        #loss = loss_point.mean()
        metrics = {
                'point_loss': loss_point.mean().item(),
                'cmd_loss': loss_cmd.mean().item(),
                'loss_steer': loss_cmd_raw[:, 0].mean().item(),
                'loss_speed': loss_cmd_raw[:, 1].mean().item(),
                'temperature': self.temperature.item()
                }

        if batch_nb % 250 == 0:
            metrics['train_image'] = viz_td(batch, points_lbc, between, out_cmd, loss_point, loss_cmd, use_wandb=self.hparams.log)
            metrics['train_heatmap'] = viz_weights(topdown, target, points_lbc, logits, loss_point, use_wandb=self.hparams.log)
            if HAS_DISPLAY:
                cv2.waitKey(1)
        if self.logger != None:
            #for _, item in metrics.items():
            #    print(_, type(item))

            self.logger.log_metrics(metrics, self.global_step)

        return {'loss': loss}

    def validation_step(self, batch, batch_nb):
        img, topdown, points, target, actions, meta = batch
        points_lbc, logits, target_heatmap = self.forward(topdown, target)

        alpha = 0.0
        between = alpha * points_lbc + (1-alpha) * points
        out_cmd = self.controller(between)
        out_cmd_pred = self.controller(points_lbc)

        loss_point = torch.nn.functional.l1_loss(points_lbc, points, reduction='none').mean((1, 2))
        loss_cmd_raw = torch.nn.functional.l1_loss(out_cmd, actions, reduction='none')
        loss_cmd_pred_raw = torch.nn.functional.l1_loss(out_cmd_pred, actions, reduction='none')

        loss_cmd = loss_cmd_raw.mean(1)
        loss = loss_point.mean()

        img = viz_td(batch, points_lbc, between, out_cmd, loss_point, loss_cmd, use_wandb=self.hparams.log)
        if batch_nb == 0 and self.logger != None:
            self.logger.log_metrics({
                'val_image': img,
                'val_heatmap': viz_weights(topdown, target, points_lbc, logits, loss_point, use_wandb=self.hparams.log)
                }, self.global_step)
            if HAS_DISPLAY:
                cv2.waitKey(1)


        result = {
                'val_loss': loss,
                'val_point_loss': loss_point.mean(),

                'val_cmd_loss': loss_cmd_raw.mean(1).mean(),
                'val_steer_loss': loss_cmd_raw[:, 0].mean(),
                'val_speed_loss': loss_cmd_raw[:, 1].mean(),

                'val_cmd_pred_loss': loss_cmd_pred_raw.mean(1).mean(),
                'val_steer_pred_loss': loss_cmd_pred_raw[:, 0].mean(),
                'val_speed_pred_loss': loss_cmd_pred_raw[:, 1].mean(),
                }

        return result

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
                optim, mode='min', factor=0.5, patience=2, min_lr=1e-6,
                verbose=True)

        return [optim], [scheduler]

    def train_dataloader(self):
        return get_dataset(self.hparams, True, self.hparams.batch_size, sample_by=self.hparams.sample_by)

    def val_dataloader(self):
        return get_dataset(self.hparams, False, self.hparams.batch_size, sample_by=self.hparams.sample_by)


def main(hparams):

    if hparams.log:
        logger = WandbLogger(id=hparams.id, save_dir=str(hparams.save_dir), project='lbc')
    else:
        logger = False
    checkpoint_callback = ModelCheckpoint(hparams.save_dir, save_top_k=3)

    if hparams.restore_from is None:
        model = MapModel(hparams)
        model_hparams = copy.copy(vars(model.hparams))
    else:
        model = MapModel.load_from_checkpoint(hparams.restore_from)
        new_hparams = vars(hparams)
        model_hparams = vars(model.hparams)
        
        print(hparams.overwrite_hparams)
        for param in hparams.overwrite_hparams + ['save_dir', 'log']:
            model_hparams[param] = new_hparams[param]
        model.hparams = dict_to_sns(model_hparams)

    # copy and postprocess for saving
    model_hparams['dataset_dir'] = str(hparams.dataset_dir)
    model_hparams['save_dir'] = str(hparams.save_dir)
    del model_hparams['data_root']
    del model_hparams['id']


    with open(hparams.save_dir / 'config.yml', 'w') as f:
            yaml.dump(model_hparams, f, default_flow_style=False, sort_keys=False)
    shutil.copyfile(hparams.dataset_dir / 'config.yml', hparams.save_dir / 'data_config.yml')

    trainer = pl.Trainer(
            gpus=hparams.gpus, max_epochs=hparams.max_epochs,
            resume_from_checkpoint=hparams.restore_from,
            logger=logger, checkpoint_callback=checkpoint_callback,
            distributed_backend='dp',)

    trainer.fit(model)

    if hparams.log:
        wandb.save(str(hparams.save_dir / '*.ckpt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-D', '--debug', action='store_true')
    parser.add_argument('-G', '--gpus', type=int, default=-1)
    parser.add_argument('--data_root', type=Path, default='/data/aaronhua')
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--steps_per_epoch', type=int, default=1000)
    parser.add_argument('--save_dir', type=Path)
    parser.add_argument('--id', type=str, default=datetime.now().strftime("%Y%m%d_%H%M%S")) 
    parser.add_argument('--log', action='store_true')
    parser.add_argument('--restore_from', type=str)
    parser.add_argument('-ow', '--overwrite_hparams', nargs='+')

    parser.add_argument('--waypoint_mode', type=str, 
            default='expectation', choices=['expectation', 'argmax'])
    parser.add_argument('--heatmap_radius', type=int, default=5)
    parser.add_argument('--sample_by', type=str, 
            default='even', choices=['none', 'even', 'speed', 'steer'])
    parser.add_argument('--command_coefficient', type=float, default=0.01)
    parser.add_argument('--temperature', type=float, default=10.0)
    parser.add_argument('--hack', action='store_true', default=True)


    # Data args.
    parser.add_argument('--dataset_dir', type=Path)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--angle_jitter', type=float, default=5)
    parser.add_argument('--pixel_jitter', type=int, default=5.5) # 3 meters


    # Optimizer args.
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.0)

    args = parser.parse_args()

    if args.dataset_dir is None:
        args.dataset_dir = Path('/data/aaronhua/leaderboard/data/lbc/autopilot/autopilot_devtest')

    suffix = f'debug/{args.id}' if args.debug else args.id
    save_dir = args.data_root / 'leaderboard/training/lbc/map_model' / suffix
    args.save_dir = save_dir
    args.save_dir.mkdir(parents=True, exist_ok=True)
    
    main(args)
