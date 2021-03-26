import copy
import argparse
import pathlib

import numpy as np
import torch
import pytorch_lightning as pl
import torchvision
import wandb
import yaml

from datetime import datetime

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from PIL import Image, ImageDraw

from lbc.carla_project.src.models import SegmentationModel, RawController
from lbc.carla_project.src.utils.heatmap import ToHeatmap
from lbc.carla_project.src.dataset import get_dataset
from lbc.carla_project.src import common


@torch.no_grad()
def visualize(batch, out, between, out_cmd, loss_point, loss_cmd, target_heatmap):
    images = list()

    for i in range(out.shape[0]):
        _loss_point = loss_point[i]
        _loss_cmd = loss_cmd[i]
        _out = out[i]
        _out_cmd = out_cmd[i]
        _between = between[i]

        rgb, topdown, points, target, actions, meta = [x[i] for x in batch]

        _rgb = np.uint8(rgb.detach().cpu().numpy().transpose(1, 2, 0) * 255)
        _target_heatmap = np.uint8(target_heatmap[i].detach().squeeze().cpu().numpy() * 255)
        _target_heatmap = np.stack(3 * [_target_heatmap], 2)
        _target_heatmap = Image.fromarray(_target_heatmap)
        _topdown = Image.fromarray(common.COLOR[topdown.argmax(0).detach().cpu().numpy()])
        _draw = ImageDraw.Draw(_topdown)

        _draw.ellipse((target[0]-2, target[1]-2, target[0]+2, target[1]+2), (255, 255, 255))

        for x, y in points:
            x = (x + 1) / 2 * 256
            y = (y + 1) / 2 * 256

            _draw.ellipse((x-2, y-2, x+2, y+2), (0, 0, 255))

        for x, y in _out:
            x = (x + 1) / 2 * 256
            y = (y + 1) / 2 * 256

            _draw.ellipse((x-2, y-2, x+2, y+2), (255, 0, 0))

        for x, y in _between:
            x = (x + 1) / 2 * 256
            y = (y + 1) / 2 * 256

            _draw.ellipse((x-1, y-1, x+1, y+1), (0, 255, 0))

        _draw.text((5, 10), 'Point: %.3f' % _loss_point)
        _draw.text((5, 30), 'Command: %.3f' % _loss_cmd)
        _draw.text((5, 50), 'Meta: %s' % meta)

        _draw.text((5, 90), 'Raw: %.3f %.3f' % tuple(actions))
        _draw.text((5, 110), 'Pred: %.3f %.3f' % tuple(_out_cmd))

        image = np.array(_topdown).transpose(2, 0, 1)
        images.append((_loss_cmd, torch.ByteTensor(image)))

    images.sort(key=lambda x: x[0], reverse=True)

    result = torchvision.utils.make_grid([x[1] for x in images], nrow=4)
    result = wandb.Image(result.numpy().transpose(1, 2, 0))

    return result


class MapModel(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams

        self.to_heatmap = ToHeatmap(hparams.heatmap_radius)
        self.net = SegmentationModel(10, 4, hparams.waypoint_mode, hack=hparams.hack, temperature=hparams.temperature)
        self.controller = RawController(4)

    def forward(self, topdown, target, debug=False):
        target_heatmap = self.to_heatmap(target, topdown)[:, None]
        out = self.net(torch.cat((topdown, target_heatmap), 1), heatmap=debug)

        if not debug:
            return out

        points, heatmap = out
        return points, (heatmap, target_heatmap)

    def training_step(self, batch, batch_nb):
        img, topdown, points, target, actions, meta = batch
        out, (heatmap, target_heatmap,) = self.forward(topdown, target, debug=True)

        alpha = torch.rand(out.shape).type_as(out)
        between = alpha * out + (1-alpha) * points
        out_cmd = self.controller(between)

        loss_point = torch.nn.functional.l1_loss(out, points, reduction='none').mean((1, 2))
        loss_cmd_raw = torch.nn.functional.l1_loss(out_cmd, actions, reduction='none')

        loss_cmd = loss_cmd_raw.mean(1)
        loss = (loss_point + self.hparams.command_coefficient * loss_cmd).mean()

        metrics = {
                'point_loss': loss_point.mean().item(),
                'cmd_loss': loss_cmd.mean().item(),
                'loss_steer': loss_cmd_raw[:, 0].mean().item(),
                'loss_speed': loss_cmd_raw[:, 1].mean().item()
                }

        if batch_nb % 250 == 0:
            metrics['train_image'] = visualize(batch, out, between, out_cmd, loss_point, loss_cmd, target_heatmap)

        if self.logger != None:
            self.logger.log_metrics(metrics, self.global_step)

        return {'loss': loss}

    def validation_step(self, batch, batch_nb):
        img, topdown, points, target, actions, meta = batch
        out, (heatmap, target_heatmap) = self.forward(topdown, target, debug=True)

        alpha = 0.0
        between = alpha * out + (1-alpha) * points
        out_cmd = self.controller(between)
        out_cmd_pred = self.controller(out)

        loss_point = torch.nn.functional.l1_loss(out, points, reduction='none').mean((1, 2))
        loss_cmd_raw = torch.nn.functional.l1_loss(out_cmd, actions, reduction='none')
        loss_cmd_pred_raw = torch.nn.functional.l1_loss(out_cmd_pred, actions, reduction='none')

        loss_cmd = loss_cmd_raw.mean(1)
        loss = (loss_point + self.hparams.command_coefficient * loss_cmd).mean()

        if batch_nb == 0 and self.logger != None:
            self.logger.log_metrics({
                'val_image': visualize(batch, out, between, out_cmd, loss_point, loss_cmd, target_heatmap)
                }, self.global_step)

        return {
                'val_loss': loss.item(),
                'val_point_loss': loss_point.mean().item(),

                'val_cmd_loss': loss_cmd_raw.mean(1).mean().item(),
                'val_steer_loss': loss_cmd_raw[:, 0].mean().item(),
                'val_speed_loss': loss_cmd_raw[:, 1].mean().item(),

                'val_cmd_pred_loss': loss_cmd_pred_raw.mean(1).mean().item(),
                'val_steer_pred_loss': loss_cmd_pred_raw[:, 0].mean().item(),
                'val_speed_pred_loss': loss_cmd_pred_raw[:, 1].mean().item(),
                }

    def validation_epoch_end(self, batch_metrics):
        results = dict()

        for metrics in batch_metrics:
            for key in metrics:
                if key not in results:
                    results[key] = list()

                results[key].append(metrics[key])

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
        return get_dataset(self.hparams.dataset_dir, True, self.hparams.batch_size, sample_by=self.hparams.sample_by)

    def val_dataloader(self):
        return get_dataset(self.hparams.dataset_dir, False, self.hparams.batch_size, sample_by=self.hparams.sample_by)


def main(hparams):
    model = MapModel(hparams)
    logger = False
    if hparams.log:
        logger = WandbLogger(id=hparams.id, save_dir=str(hparams.save_dir), project=f'lbc_{hparams.waypoint_mode}')
    checkpoint_callback = ModelCheckpoint(hparams.save_dir, save_top_k=1)

    try:
        resume_from_checkpoint = sorted(hparams.save_dir.glob('*.ckpt'))[-1]
    except:
        resume_from_checkpoint = None

    trainer = pl.Trainer(
            gpus=-1, max_epochs=hparams.max_epochs,
            resume_from_checkpoint=resume_from_checkpoint,
            logger=logger, checkpoint_callback=checkpoint_callback)


    with open(hparams.save_dir / 'config.yml', 'w') as f:
        hparams_copy = copy.copy(vars(model.hparams))
        hparams_copy['dataset_dir'] = str(hparams.dataset_dir)
        hparams_copy['save_dir'] = str(hparams.save_dir)
        del hparams_copy['data_root']
        yaml.dump(hparams_copy, f, default_flow_style=False, sort_keys=False)

    trainer.fit(model)

    if hparams.log:
        wandb.save(str(hparams.save_dir / '*.ckpt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-D', '--debug', action='store_true')
    parser.add_argument('-G', '--gpus', type=int, default=-1)
    parser.add_argument('--data_root', type=pathlib.Path, default='/data')
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--save_dir', type=pathlib.Path)
    parser.add_argument('--id', type=str, default=datetime.now().strftime("%Y%m%d_%H%M%S")) 
    parser.add_argument('--log', action='store_true')

    parser.add_argument('--waypoint_mode', type=str, default='expectation', choices=['expectation', 'argmax'])
    parser.add_argument('--heatmap_radius', type=int, default=5)
    parser.add_argument('--sample_by', type=str, choices=['none', 'even', 'speed', 'steer'], default='even')
    parser.add_argument('--command_coefficient', type=float, default=0.1)
    parser.add_argument('--temperature', type=float, default=10.0)
    parser.add_argument('--hack', action='store_true', default=False)

    # Data args.
    parser.add_argument('--dataset_dir', type=pathlib.Path, required=True)
    parser.add_argument('--batch_size', type=int, default=4)

    # Optimizer args.
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.0)

    parsed = parser.parse_args()
    suffix = f'debug/{parsed.id}' if parsed.debug else parsed.id
    save_dir = parsed.data_root / 'leaderboard/training/lbc/map_model' / parsed.waypoint_mode / suffix
    parsed.save_dir = save_dir
    parsed.save_dir.mkdir(parents=True, exist_ok=True)


    main(parsed)