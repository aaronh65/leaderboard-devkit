import os, argparse, copy, shutil, yaml, time
from pathlib import Path
from datetime import datetime

from PIL import Image, ImageDraw
from matplotlib import cm
import cv2, numpy as np

import torch, torchvision, wandb
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from misc.utils import *
from dqn.src.agents.models import SegmentationModel, DiscreteController, SpatialSoftmax
from dqn.src.agents.heatmap import ToHeatmap, ToTemporalHeatmap
from lbc.carla_project.src.common import CONVERTER, COLOR
from lbc.carla_project.src.map_model import viz_weights
 
HAS_DISPLAY = int(os.environ['HAS_DISPLAY'])
PRIORITY = False
#from dqn.src.offline.dataset import get_dataloader
from dqn.src.offline.split_dataset import get_dataloader

text_color = (255,255,255)
aim_color = (60,179,113) # dark green
student_color = (65,105,225) # dark blue
expert_color = (178,34,34) # dark red
route_colors = [(255,255,255), (112,128,144), (47,79,79), (47,79,79)] 

def transform_action(action, hparams, to_flat=False, to_spatial=False):
    assert not (to_flat and to_spatial)
    assert to_flat or to_spatial
    if to_spatial: # action is N,nSp*nSt -> N,1,nSp,nSt
        action = action.view((-1, hparams.n_throttle, hparams.n_steer)).unsqueeze(1)
    else: # action is N,1,nSp,nSt -> N,nSp*nSt
        action = action.squeeze(1).view((-1, hparams.n_throttle*hparams.n_steer))
    return action

# takes (N,C,H,W) topdown and (N,4,H,W) logits
# averages t=0.5s,1.0s logitss and overlays it on topdown
@torch.no_grad()
def fuse_logits(topdown, logits, alpha=0.5):
    logits_norm = spatial_norm(logits).cpu().numpy() #N,T,H,W
    fused = np.array(COLOR[topdown.argmax(1).cpu()]).astype(np.uint8) # N,H,W,3
    for i in range(fused.shape[0]):
        map1 = cm.inferno(logits_norm[i][0])[...,:3]*255
        map2 = cm.inferno(logits_norm[i][1])[...,:3]*255
        logits_mask = cv2.addWeighted(np.uint8(map1), 0.5, np.uint8(map2), 0.5, 0)
        fused[i] = cv2.addWeighted(logits_mask, alpha, fused[i], 1, 0)
    fused = fused.astype(np.uint8) # (N,H,W,3)
    return fused

@torch.no_grad()
def visualize(batch, meta):

    _, action, reward, ntopdown, done, info = batch
    topdown, tmap, logits = meta['topdown'], meta['tmap'], meta['logits']
    pts_em, ctrl_em, ctrl_e = meta['pts_em'], meta['ctrl_em'], meta['ctrl_e']
    pts_pm, ctrl_pm, ctrl_p = meta['pts_pm'], meta['ctrl_pm'], meta['ctrl_p']
    point_loss, margin_loss = meta['point_loss'], meta['margin_loss'],
    margin_map = meta['margin_map']
    Q_ctrl_pm = meta['Q_ctrl_pm']

    tdown = fuse_logits(topdown, logits)
    #tdown = COLOR[topdown.argmax(1).cpu()]
    #tdown[tmap.squeeze(1).cpu() > 0.1] = 255
    Q_norm = spatial_norm(Q_ctrl_pm).cpu().numpy()
    Q_pred = spatial_select(Q_ctrl_pm, ctrl_pm).squeeze(-1)
    Q_exp  = spatial_select(Q_ctrl_pm, ctrl_em.unsqueeze(1)).squeeze(-1)
    _ctrl_p = ctrl_p.squeeze(1).cpu()
    _ctrl_pm = ctrl_pm.squeeze(1).cpu()

    images = list()
    indices = np.argsort(margin_loss.detach().cpu().numpy().flatten())
    indices = list(reversed(indices))[:8]
    n,c,h,w = topdown.shape
    for i in indices:

        _tdown = Image.fromarray(tdown[i])
        _draw = ImageDraw.Draw(_tdown)
        for x,y in pts_em[i]:
            _draw.ellipse((x-2, y-2, x+2, y+2), expert_color)
        for x,y in pts_pm[i]:
            _draw.ellipse((x-2, y-2, x+2, y+2), student_color)

        _metadata = decode_str(info['metadata'][i])
        _draw.text((5,10), f'meta = {_metadata}', text_color)
        _draw.text((5,20), 
                f'points/margin loss:   {point_loss[i]:.3f}/{margin_loss[i]:.3f}', text_color)
        _draw.text((5,30), 
                f'student/expert value: {Q_exp[i].item():.3f}/{Q_pred[i].item():.3f}', text_color)
        steer_e, throttle_e = ctrl_e[i]
        _draw.text((5,40), f'expert action:  {steer_e:.3f}/{throttle_e:.3f}', text_color)
        steer_p, throttle_p = _ctrl_p[i]
        _draw.text((5,50), f'student action: {steer_p:.3f}/{throttle_p:.3f}', text_color)
        _draw.text((5,60), f'reward = {reward[i].item():.3f}', text_color)
        _tdown = np.array(_tdown)
        #_tdown = cv2.cvtColor(np.array(_tdown), cv2.COLOR_RGB2BGR)

        _Q_image = Q_norm[i].transpose(1,2,0)
        _Q_image = np.uint8(_Q_image*255)
        _Q_image = np.tile(_Q_image, (1,1,3))
        x,y = np.rint(ctrl_em[i].detach().cpu().numpy()).astype(int)
        _Q_image[y,x] = expert_color
        x,y = np.rint(_ctrl_pm[i].detach().cpu().numpy()).astype(int)
        _Q_image[y,x] = student_color
        _Q_image = np.array(_Q_image)

        _margin_map = margin_map[i].cpu().numpy()
        _margin_map = np.uint8(_margin_map*255)
        _margin_map = np.expand_dims(_margin_map,-1)
        _margin_map = np.tile(_margin_map, (1,1,3))

        image = np.vstack((_Q_image, _margin_map))
        image = Image.fromarray(image)
        image = np.array(image.resize((h,w), resample=0))
        image = np.hstack((_tdown, image))

        
        images.append(torch.ByteTensor(image.transpose(2,0,1)))
    images = torchvision.utils.make_grid(images, nrow=2)
    images = images.numpy().transpose(1,2,0)
    return images

# just needs to know if it's rolling out or nah
class MapModel(pl.LightningModule):
    def __init__(self, hparams=None):
        super().__init__()

        self.hparams = hparams
        self.to_heatmap = ToHeatmap(hparams.expert_radius)
        self.register_buffer('temperature', torch.Tensor([hparams.temperature]))
        self.net = SegmentationModel(10, 4, batch_norm=True, hack=hparams.hack, extract=False)

        self.controller = DiscreteController(n_input=4)
        self.td_criterion = torch.nn.MSELoss(reduction='none')
        self.margin_criterion = torch.nn.MSELoss(reduction='none')
        
    def forward(self, topdown, target, debug=False):
        target_heatmap = self.to_heatmap(target, topdown)[:, None]
        input = torch.cat((topdown, target_heatmap), 1)
        # points (N,T,2), logits/weights (N,T,H,W)
        points, logits, weights = self.net(input, temperature=self.temperature.item())
        return points, logits, target_heatmap

    # two modes: sample, argmax?
    def get_argmax_actions(self, Qmap):
        h,w = Qmap.shape[-2:]
        Qmap_flat = Qmap.view(Qmap.shape[:-2] + (-1,)) # (N,T,H*W)
        Q_all, action_flat = torch.max(Qmap_flat, -1, keepdim=True) # (N,T,1)

        action = torch.cat((
            action_flat % w,
            action_flat // w),
            axis=2) # (N,T,2)
        
        return action, Q_all # (N,T,2), (N,T,1)

    def on_epoch_start(self):
        self.train_data.dataset.epoch_num = self.current_epoch
        self.val_data.dataset.epoch_num = self.current_epoch

    def training_step(self, batch, batch_nb):
        state, action, reward, next_state, done, info = batch
        topdown, target = state
        points_expert, ctrl_expert = action

        points_pred, logits, tmap = self.forward(topdown, target, debug=True)
        point_loss = torch.nn.functional.l1_loss(points_pred, points_expert, reduction='none')
        point_loss = point_loss.mean((-1,-2))

        # Q_ctrl_pm = "Q map for control output given predicted points"
        Q_ctrl_pm = self.controller(points_pred)
        Q_ctrl_pm = transform_action(Q_ctrl_pm, self.hparams, to_spatial=True) #N,1,nSp,nSt
        ctrl_pm, Q_ctrl_p = self.get_argmax_actions(Q_ctrl_pm) # ctrl in map space
        ctrl_pm = ctrl_pm.float() #N,1,2

        # convert expert control to map space
        # x (steer) from (-1,1) to [0,20]
        # y (throttle) from (0,1) to [0,10]
        ctrl_e = ctrl_expert
        ctrl_em = ctrl_e.clone()
        ctrl_em[:,0] = (ctrl_em[:,0]+1) / 2 * (self.hparams.n_steer-1) 
        ctrl_em[:,1] = (1-ctrl_em[:,1]) * (self.hparams.n_throttle-1)

        ## Q_ctrl_bm = "Q map for control output given between points"
        #alpha = torch.rand(points.shape).type_as(points)
        #between = alpha * points + (1-alpha) * points_expert
        #Q_ctrl_bm = self.controller(between)
        #Q_ctrl_bm = transform_action(Q_ctrl_bm, self.hparams, to_spatial=True) #N,1,nSp,nSt
        #ctrl_b, Q_ctrl_b = self.get_argmax_actions(Q_ctrl_bm) # this is in pixel space

        #out_cmd_pred = self.controller(points_lbc)


        #Q_ctrl_e = spatial_select(Q_ctrl_pm, points_pred.unsqueeze(1))
        Q_exp  = spatial_select(Q_ctrl_pm, ctrl_em.unsqueeze(1)).squeeze(-1)
        
        margin_switch = info['margin_switch']
        margin_map = self.to_heatmap(ctrl_em, Q_ctrl_pm) #N,nSp,nSt
        margin_map = (1 - margin_map) * self.hparams.expert_margin
        margin = Q_ctrl_pm.squeeze(1) + margin_map - Q_exp.unsqueeze(-1)
        margin = F.relu(margin)
        margin_loss = self.hparams.lambda_margin * margin.mean((-1,-2))

        td_loss = self.hparams.lambda_td * 0
        batch_loss = td_loss + margin_loss + point_loss

        metrics = {}
        metrics['train_point_loss'] = point_loss.mean().item()
        metrics['train_margin_loss'] = margin_loss.mean().item()
        if batch_nb == 0:
            metrics['epochs'] = self.current_epoch
            metrics['hard_prop'] = info['hard_prop'].mean().item()

        if batch_nb % 250 == 0 and (self.logger != None or HAS_DISPLAY):
            # convert predicted control to control space
            # x (steer) from [0,20] to (-1,1)
            # y (throttle) from [0,10] to [0,1]
            ctrl_p = ctrl_pm.clone().detach().squeeze(1)
            ctrl_p[:,0] = ctrl_p[:,0] / (self.hparams.n_steer-1) * 2 - 1
            ctrl_p[:,1] = ctrl_p[:,1] / (self.hparams.n_throttle-1)
            ctrl_p[:,1] = (1-ctrl_p[:,1])

            meta = {
                'topdown': topdown,
                'tmap': tmap.squeeze(1),
                'logits': logits,
                'pts_em': (points_expert.clone()+1)/2*255,
                'pts_pm': (points_pred.clone()+1)/2*255,
                'Q_ctrl_pm': Q_ctrl_pm,
                'ctrl_e': ctrl_e,
                'ctrl_em': ctrl_em,
                'ctrl_p': ctrl_p,
                'ctrl_pm': ctrl_pm,
                'point_loss': point_loss,
                'margin_loss': margin_loss,
                'margin_map': margin_map,
            }


            images = visualize(batch, meta)
            metrics['train_image'] = wandb.Image(images)
            if HAS_DISPLAY:
                images = cv2.cvtColor(images, cv2.COLOR_RGB2BGR)
                cv2.imshow('debug', images)
                cv2.waitKey(100)

        if self.logger != None:
            self.logger.log_metrics(metrics, self.global_step)

        loss = torch.mean(batch_loss, dim=0, keepdim=True)
        return {'loss': loss}

    # make this a validation episode rollout?
    def validation_step(self, batch, batch_nb):

        state, action, reward, next_state, done, info = batch
        topdown, target = state
        points_expert, ctrl_expert = action
        with torch.no_grad():
            points_pred, logits, tmap = self.forward(topdown, target, debug=True)

        point_loss = torch.nn.functional.l1_loss(points_pred, points_expert, reduction='none')
        point_loss = point_loss.mean((-1,-2))

        # Q_ctrl_pm = "Q map for control output given predicted points"
        with torch.no_grad():
            Q_ctrl_pm = self.controller(points_pred)
        Q_ctrl_pm = transform_action(Q_ctrl_pm, self.hparams, to_spatial=True) #N,1,nSp,nSt
        ctrl_pm, Q_ctrl_p = self.get_argmax_actions(Q_ctrl_pm) # ctrl in map space
        ctrl_pm = ctrl_pm.float() #N,1,2

        # convert expert control to map space
        # x (steer) from (-1,1) to [0,20]
        # y (throttle) from (0,1) to [0,10]
        ctrl_e = ctrl_expert
        ctrl_em = ctrl_e.clone()
        ctrl_em[:,0] = (ctrl_em[:,0]+1) / 2 * (self.hparams.n_steer-1) 
        ctrl_em[:,1] = (1-ctrl_em[:,1]) * (self.hparams.n_throttle-1)

        ## Q_ctrl_bm = "Q map for control output given between points"
        #alpha = torch.rand(points.shape).type_as(points)
        #between = alpha * points + (1-alpha) * points_expert
        #Q_ctrl_bm = self.controller(between)
        #Q_ctrl_bm = transform_action(Q_ctrl_bm, self.hparams, to_spatial=True) #N,1,nSp,nSt
        #ctrl_b, Q_ctrl_b = self.get_argmax_actions(Q_ctrl_bm) # this is in pixel space

        #out_cmd_pred = self.controller(points_lbc)

        #Q_ctrl_e = spatial_select(Q_ctrl_pm, points_pred.unsqueeze(1))
        Q_exp  = spatial_select(Q_ctrl_pm, ctrl_em.unsqueeze(1)).squeeze(-1)
        
        margin_map = self.to_heatmap(ctrl_em, Q_ctrl_pm) #N,nSp,nSt
        margin_map = (1 - margin_map) * self.hparams.expert_margin
        margin = Q_ctrl_pm.squeeze(1) + margin_map - Q_exp.unsqueeze(-1)
        margin = F.relu(margin)
        margin_loss = self.hparams.lambda_margin * margin.mean((-1,-2))
                
        td_loss = self.hparams.lambda_td * 0
        batch_loss = td_loss + margin_loss + point_loss

        metrics = {}
        #metrics['val_loss'] = batch_loss.mean().item()
        if batch_nb == 0 and (self.logger != None or HAS_DISPLAY):

            # convert predicted control to control space
            # x (steer) from [0,20] to (-1,1)
            # y (throttle) from [0,10] to [0,1]
            ctrl_p = ctrl_pm.clone().detach().squeeze(1)
            ctrl_p[:,0] = ctrl_p[:,0] / (self.hparams.n_steer-1) * 2 - 1
            ctrl_p[:,1] = ctrl_p[:,1] / (self.hparams.n_throttle-1)
            ctrl_p[:,1] = (1-ctrl_p[:,1])

            meta = {
                'topdown': topdown,
                'tmap': tmap.squeeze(1),
                'logits': logits,
                'pts_em': (points_expert.clone()+1)/2*255,
                'pts_pm': (points_pred.clone()+1)/2*255,
                'Q_ctrl_pm': Q_ctrl_pm,
                'ctrl_e': ctrl_e,
                'ctrl_em': ctrl_em,
                'ctrl_p': ctrl_p,
                'ctrl_pm': ctrl_pm,
                'point_loss': point_loss,
                'margin_loss': margin_loss,
                'margin_map': margin_map,
            }

            images = visualize(batch, meta)
            metrics['val_image'] = wandb.Image(images)
            if HAS_DISPLAY:
                images = cv2.cvtColor(images, cv2.COLOR_RGB2BGR)
                cv2.imshow('debug', images)
                cv2.waitKey(100)

        if self.logger != None:
            self.logger.log_metrics(metrics, self.global_step)

        val_loss =  torch.mean(batch_loss, dim=0, keepdim=True)
        val_margin_loss = torch.mean(margin_loss, dim=0, keepdim=True)
        val_point_loss = torch.mean(point_loss, dim=0, keepdim=True)
        return {'val_loss': val_loss,
                'val_margin_loss': val_margin_loss,
                'val_point_loss': val_point_loss,
                }

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
                list(self.net.parameters()) + list(self.controller.parameters()) + list(self.parameters()),
                lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optim, mode='min', factor=0.5, patience=5, min_lr=1e-6,
                verbose=True)
        return [optim], [scheduler]

    def train_dataloader(self):

        self.train_data = get_dataloader(self.hparams,self.hparams.train_dataset,is_train=True)
        return self.train_data

    # online val dataloaders spoof a length of N batches, and do N episode rollouts
    def val_dataloader(self):

        self.val_data = get_dataloader(self.hparams,self.hparams.val_dataset,is_train=False)
        return self.val_data


# offline training

def main(hparams):
    
    if hparams.log:
        logger = WandbLogger(id=hparams.id, save_dir=str(hparams.save_dir), project='dqn_offline')
    else:
        logger = False
    checkpoint_callback = ModelCheckpoint(hparams.save_dir, save_top_k=3, verbose=True, period=10)

    # resume and add a couple arguments
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
    model_hparams['train_dataset'] = str(hparams.train_dataset)
    model_hparams['val_dataset'] = str(hparams.val_dataset)
    model_hparams['save_dir'] = str(hparams.save_dir)
    del model_hparams['data_root']
    del model_hparams['id']

    with open(hparams.save_dir / 'config.yml', 'w') as f:
        yaml.dump(model_hparams, f, default_flow_style=False, sort_keys=False)
    shutil.copyfile(hparams.train_dataset / 'config.yml', hparams.save_dir / 'train_data_config.yml')
    shutil.copyfile(hparams.val_dataset / 'config.yml', hparams.save_dir / 'val_data_config.yml')

    # offline trainer can use all gpus
    # when resuming, the network starts at epoch 36
    trainer = pl.Trainer(
        gpus=hparams.gpus, 
        max_epochs=hparams.max_epochs,
        logger=logger,
        checkpoint_callback=checkpoint_callback,
        distributed_backend='dp',)

    trainer.fit(model)

    checkpoint_callback._save_model(str(hparams.save_dir / 'last.ckpt'))

    if hparams.log:
        wandb.save(str(hparams.save_dir / '*.ckpt'))




if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('-D', '--debug', action='store_true')
    parser.add_argument('-G', '--gpus', nargs='+', type=int, default=[-1])
    parser.add_argument('--restore_from', type=str)
    parser.add_argument('--save_dir', type=Path)
    parser.add_argument('--id', type=str, default=datetime.now().strftime("%Y%m%d_%H%M%S")) 
    parser.add_argument('--log', action='store_true')
    parser.add_argument('-ow', '--overwrite_hparams', nargs='+')

    # Trainer args
    parser.add_argument('--train_dataset', type=Path)
    parser.add_argument('--val_dataset', type=Path)
    parser.add_argument('--hard_prop', type=float, default=0.0)
    parser.add_argument('--max_prop_epoch', type=int)
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lambda_margin', type=float, default=1.0)
    parser.add_argument('--lambda_td', type=float, default=1.0)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--n', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--sample_by', type=str, 
            choices=['none', 'even', 'speed', 'steer'], default='even')
    parser.add_argument('--weight_decay', type=float, default=0.0)

    # Model args
    parser.add_argument('--n_steer', type=int, default=21)
    parser.add_argument('--n_throttle', type=int, default=11)
    parser.add_argument('--throttle_mode', type=str, default='speed', 
            choices=['speed', 'throttle'])
    parser.add_argument('--max_speed', type=int, default=10)
    parser.add_argument('--heatmap_radius', type=int, default=5)
    parser.add_argument('--expert_radius', type=int, default=2)
    parser.add_argument('--expert_margin', type=float, default=1.0)
    parser.add_argument('--temperature', type=float, default=10.0)
    parser.add_argument('--hack', action='store_true', default=True)
    parser.add_argument('--control_type', type=str, default='learned')
        
    args = parser.parse_args()

    if args.train_dataset is None:
        args.train_dataset = Path('/data/aaronhua/leaderboard/data/lbc/autopilot/autopilot_devtest')
        #args.train_dataset = Path('/data/aaronhua/leaderboard/data/lbc/autopilot/autopilot_devtest_toy')
    if args.val_dataset is None:
        args.val_dataset = Path('/data/aaronhua/leaderboard/data/lbc/autopilot/autopilot_devtest')
        #args.val_dataset = Path('/data/aaronhua/leaderboard/data/lbc/autopilot/autopilot_devtest_toy')
    if args.gpus[0] == -1:
        args.gpus = -1

    _, drive, name = str(args.train_dataset).split('/')[:3]
    args.data_root = Path(f'/{drive}', name)
    suffix = f'debug/{args.id}' if args.debug else args.id
    save_dir = args.data_root / f'leaderboard/training/dqn/offline/{suffix}'
    args.save_dir = save_dir
    args.save_dir.mkdir(parents=True, exist_ok=True)

    main(args)


