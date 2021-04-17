import os, argparse, copy, shutil, yaml
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
from dqn.src.agents.models import SegmentationModel, RawController, SpatialSoftmax
from dqn.src.agents.heatmap import ToHeatmap, ToTemporalHeatmap
from dqn.src.offline.dataset import get_dataloader
from lbc.carla_project.src.common import CONVERTER, COLOR
from lbc.carla_project.src.map_model import plot_weights
 
HAS_DISPLAY = int(os.environ['HAS_DISPLAY'])
text_color = (255,255,255)
aim_color = (60,179,113) # dark green
student_color = (65,105,225) # dark blue
expert_color = (178,34,34) # dark red
route_colors = [(255,255,255), (112,128,144), (47,79,79), (47,79,79)] 


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
def viz_Qmap(batch, meta, alpha=0.5, r=2):
    state, action, reward, next_state, done, info = batch
    topdown, target = state
    points_expert = info['points_expert']

    Qmap, Q_expert = meta['Qmap'], meta['Q_expert']
    batch_loss, td_loss, margin = meta['batch_loss'], meta['td_loss'], meta['margin']
    margin = meta['margin']

    N,T,H,W = Qmap.shape
    Qmap_norm = spatial_norm(Qmap).cpu().numpy() #N,T,H,W
    fused = COLOR[topdown.argmax(1).cpu()]
    fused = np.array(fused).astype(np.uint8) # N,H,W,3
    fused = np.expand_dims(fused,axis=1) # N,1,H,W,3
    fused = np.tile(fused, (1,T,1,1,1)) # N,T,H,W,3

    Q_mins, points_min = torch.min(Qmap.view(N,T,H*W), dim=-1)
    Q_maxs, points_max = torch.max(Qmap.view(N,T,H*W), dim=-1)

    _margin = margin.mean(1).flatten().detach().clone().cpu().numpy()
    indices = np.argsort(_margin)[:8]
    for n in indices:
        tensor = Qmap_norm[n]
        for t, hmap in enumerate(tensor):
            hmap = np.uint8(cm.inferno(hmap)[...,:3]*255)
            img = cv2.addWeighted(hmap, alpha, fused[n][t], 1, 0)

            img = Image.fromarray(img)
            draw = ImageDraw.Draw(img)
            x, y = points_max[n][t] % W, points_max[n][t] // W
            draw.ellipse((x-r,y-r,x+r,y+r), (0,0,255))
            if 'points_expert' in info.keys():
                x, y = points_expert[n][t]
                draw.ellipse((x-r,y-r,x+r,y+r), expert_color)
            x, y = points_min[n][t] % W, points_min[n][t] // W
            draw.ellipse((x-r,y-r,x+r,y+r), (255,255,255))
            draw.text((5,10), f'Q_max: \t{Q_maxs[n][t]:.2f}', text_color)
            draw.text((5,20), f'Q_exp: \t{Q_expert[n][t]:.2f}', text_color)
            draw.text((5,30), f'Q_min: \t{Q_mins[n][t]:.2f}', text_color)
            draw.text((5,40), f'loss: \t{margin[n][t]:.2f}', text_color)

            #img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
            fused[n][t] = np.array(img)

    result = [np.hstack(fused[n]) for n in indices]
    result = np.vstack(result)
    return result

# visualize each timestep's heatmap?
@torch.no_grad()
def viz_td(batch, meta, alpha=0.5, r=2):

    Q, nQ = meta['Q'], meta['nQ']
    Qmap, nQmap = meta['Qmap'], meta['nQmap']
    tmap, ntmap = meta['tmap'], meta['ntmap']
    naction = meta['naction']
    hparams = meta['hparams']
    batch_loss, td_loss, margin = meta['batch_loss'], meta['td_loss'], meta['margin']
    margin_loss = margin.mean(1).detach().cpu().numpy()


    state, action, reward, next_state, done, info = batch
    topdown, target = state
    fused = fuse_logits(topdown, Qmap)
    ntopdown, ntarget = next_state
    nfused = fuse_logits(ntopdown, nQmap)
    discount = info['discount'].cpu()

    images = list()
    indices = np.argsort(batch_loss.clone().detach().cpu().numpy().flatten())[::-1]
    indices = indices[:16]

    for i in indices:

        # current state
        _topdown = fused[i].copy()
        _topdown[tmap[i][0].cpu() > 0.1] = 255
        _topdown = Image.fromarray(_topdown)
        _draw = ImageDraw.Draw(_topdown)
        _action = action[i].cpu().numpy().astype(np.uint8) # (4,2)
        for x, y in _action:
            _draw.ellipse((x-r, y-r, x+r, y+r), expert_color)
        x, y = np.mean(_action[0:2], axis=0)
        _draw.ellipse((x-r, y-r, x+r, y+r), aim_color)

        _draw.text((5, 10), f'Q = {Q[i].item():.2f}', text_color)
        _draw.text((5, 20), f'reward = {reward[i].item():.3f}', text_color)
        _draw.text((5, 30), f'discount = {discount[i].item():.2f}', text_color)
        _draw.text((5, 40), f'nQ = {nQ[i].item():.2f}', text_color)


        # next state
        _ntopdown = nfused[i].copy()
        _ntopdown[ntmap[i][0].cpu() > 0.1] = 255
        _ntopdown = Image.fromarray(_ntopdown)
        _ndraw = ImageDraw.Draw(_ntopdown)
        _naction = naction[i].cpu().numpy().astype(np.uint8) # (4,2)
        for x, y in _naction:
            _ndraw.ellipse((x-r, y-r, x+r, y+r), student_color)
        x, y = np.mean(_naction[0:2], axis=0)
        _ndraw.ellipse((x-r, y-r, x+r, y+r), aim_color)

        _metadata = decode_str(info['metadata'][i])
        _ndraw.text((5, 10), f'meta = {_metadata}', text_color)
        _ndraw.text((5, 20), f'done = {bool(done[i])}', text_color)
        _td_loss = f'{td_loss[i].item():.2f}'
        _margin_loss = f'{margin_loss[i].mean().item():.2f}'
        _batch_loss = f'{batch_loss[i].item():.2f}'
        _ndraw.text((5, 30), f'batch_loss = {_td_loss} + {_margin_loss} = {_batch_loss}', text_color)
#
        _combined = np.hstack((np.array(_topdown), np.array(_ntopdown)))
        #_combined = cv2.cvtColor(_combined, cv2.COLOR_BGR2RGB)
        images.append(_combined)
    
    result = [torch.ByteTensor(x.transpose(2,0,1)) for x in images]
    result = torchvision.utils.make_grid(result, nrow=2)
    result = result.numpy().transpose(1,2,0)
    return result

@torch.no_grad()
def make_histogram(tensor, in_type, b_i=0, c_i=0):
    data = tensor.detach().cpu().numpy()[b_i][c_i].flatten()
    if len(data) > 10000:
        indices = np.random.randint(0, data.shape[0], 10000)
        data = data[indices]
    hist = wandb.Histogram(data)
    return hist

# just needs to know if it's rolling out or nah
class MapModel(pl.LightningModule):
    def __init__(self, hparams=None):
        super().__init__()

        if hparams is not None:
            self.hparams = hparams
            self.to_heatmap = ToHeatmap(hparams.heatmap_radius)
            self.expert_heatmap = ToTemporalHeatmap(hparams.expert_radius)
            self.register_buffer('temperature', torch.Tensor([hparams.temperature]))
            self.net = SegmentationModel(10, 4, batch_norm=True, hack=hparams.hack)

        self.controller = RawController(4)
        self.td_criterion = torch.nn.MSELoss(reduction='none')
        self.margin_criterion = torch.nn.MSELoss(reduction='none')

    def restore_from_lbc(self, weights_path):
        from lbc.carla_project.src.map_model import MapModel as LBCModel

        print('restoring from LBC')
        lbc_model = LBCModel.load_from_checkpoint(weights_path)
        self.net = lbc_model.net
        self.temperature = lbc_model.temperature
        self.to_heatmap = lbc_model.to_heatmap

    def forward(self, topdown, target, debug=False):
        target_heatmap = self.to_heatmap(target, topdown)[:, None]
        input = torch.cat((topdown, target_heatmap), 1)
        # points (N,T,2), logits/weights (N,T,H,W)
        points, logits, weights = self.net(input, temperature=self.temperature.item())
        Qmap = logits
        if debug:
            return points, logits, weights, target_heatmap, Qmap
        else:
            return points, target_heatmap, Qmap

        # two modes: sample, argmax?
    def get_dqn_actions(self, Qmap, explore=False):
        h,w = Qmap.shape[-2:]
        Qmap_flat = Qmap.view(Qmap.shape[:-2] + (-1,)) # (N,T,H*W)
        Q_all, action_flat = torch.max(Qmap_flat, -1, keepdim=True) # (N,T,1)

        action = torch.cat((
            action_flat % w,
            action_flat // w),
            axis=2) # (N,T,2)
        
        return action, Q_all # (N,T,2), (N,T,1)

    def training_step(self, batch, batch_nb):
        metrics ={}

        state, action, reward, next_state, done, info = batch
        topdown, target = state
        points, logits, weights, tmap, Qmap = self.forward(topdown, target, debug=True)
        Q_all = spatial_select(Qmap, action)
        Q = torch.mean(Q_all, axis=1, keepdim=False)

        ntopdown, ntarget = next_state
        with torch.no_grad():
            npoints, nlogits, nweights, ntmap, nQmap = self.forward(ntopdown, ntarget, debug=True)
        naction, nQ_all = self.get_dqn_actions(nQmap)
        nQ = torch.mean(nQ_all, axis=1, keepdim=False)

        # td loss
        discount = info['discount']
        td_target = reward + discount * nQ * (1-done)
        td_loss = self.hparams.lambda_td * self.td_criterion(Q, td_target) # TD(n) error Nx1

        # expert margin loss
        points_expert = info['points_expert']
        Q_expert_all = spatial_select(Qmap, info['points_expert']) #N,T,1

        margin_map = self.expert_heatmap(points_expert, Qmap) #[0,1] tall at expert points
        margin_map = (1-margin_map)*self.hparams.expert_margin
        margin = Qmap + margin_map - Q_expert_all.unsqueeze(-1)
        margin = F.relu(margin)
        margin = torch.mean(margin, dim=(-1,-2)) #N,T
        margin_loss = self.hparams.lambda_margin * margin.mean(dim=1,keepdim=True)

        batch_loss = td_loss + margin_loss #N,1
        loss = torch.mean(batch_loss, dim=0) #1,

        if batch_nb % 250 == 0:
            meta = {
                'hparams': self.hparams,
                'Qmap': Qmap, 'nQmap': nQmap, 
                'Q': Q, 'nQ': nQ, 'Q_expert': Q_expert_all.squeeze(dim=-1),
                'naction': naction,
                'tmap': tmap, 'ntmap': ntmap,
                'batch_loss': batch_loss,
                'td_loss': td_loss,
                'margin': margin,
                }
            vQmap, vtd = viz_Qmap(batch, meta), viz_td(batch, meta)
            if HAS_DISPLAY:
                cv2.imshow('td', cv2.cvtColor(vtd, cv2.COLOR_BGR2RGB))
                cv2.imshow('Qmap', cv2.cvtColor(vQmap, cv2.COLOR_BGR2RGB))
                cv2.waitKey(5000)

        if self.logger != None:
            metrics = {
                        f'train/TD({self.hparams.n}) loss': td_loss.mean().item(),
                        'train/margin_loss': margin_loss.mean().item(),
                        'train/batch_loss': batch_loss.mean().item(),
                        }

            if batch_nb % 250 == 0:
                metrics['train_td'] = wandb.Image(vtd)
                metrics['train_Qmap'] = wandb.Image(vQmap)
                to_hist = {'Qmap': Qmap}
                for key, item in to_hist.items():
                    metrics[f'train/{key}_hist'] = make_histogram(item, key)
            self.logger.log_metrics(metrics, self.global_step)

        return {'loss': loss}

    # make this a validation episode rollout?
    def validation_step(self, batch, batch_nb):
        metrics = {}

        state, action, reward, next_state, done, info = batch
        topdown, target = state
        with torch.no_grad():
            points, logits, weights, tmap, Qmap = self.forward(topdown, target, debug=True)
        Q_all = spatial_select(Qmap, action).squeeze() #NxT
        Q = torch.mean(Q_all, axis=-1, keepdim=True)

        ntopdown, ntarget = next_state
        with torch.no_grad():
            npoints, nlogits, nweights, ntmap, nQmap = self.forward(ntopdown, ntarget, debug=True)
        naction, nQ_all = self.get_dqn_actions(nQmap)
        nQ = torch.mean(nQ_all.squeeze(), axis=-1, keepdim=True) # N,1

        # td loss
        discount = info['discount']
        td_target = reward + discount * nQ * (1-done)
        td_loss = self.hparams.lambda_td * self.td_criterion(Q, td_target) # TD(n) error Nx1

        # expert margin loss
        points_expert = info['points_expert']
        Q_expert_all = spatial_select(Qmap, points_expert) #N,T,1

        # DEBUG

        margin_map = self.expert_heatmap(points_expert, Qmap) #[0,1] tall at expert points
        margin_map = (1-margin_map)*self.hparams.expert_margin #[0, 8] low at expert points
        margin = Qmap + margin_map - Q_expert_all.unsqueeze(-1)
        margin = F.relu(margin)
        margin = torch.mean(margin, dim=(-1,-2)) #N,T
        margin_loss = self.hparams.lambda_margin * margin.mean(dim=1,keepdim=True)

        batch_loss = td_loss + margin_loss
        val_loss = torch.mean(batch_loss, axis=0)
        
        if batch_nb == 0 and self.logger != None:
            meta = {
                'hparams': self.hparams,
                'Qmap': Qmap, 'nQmap': nQmap, 
                'Q': Q, 'nQ': nQ, 'Q_expert': Q_expert_all.squeeze(dim=-1),
                'naction': naction,
                'tmap': tmap, 'ntmap': ntmap,
                'batch_loss': batch_loss,
                'td_loss': td_loss,
                'margin': margin,
                }
            vQmap, vtd = viz_Qmap(batch, meta), viz_td(batch, meta)
            if HAS_DISPLAY:
                cv2.imshow('td', cv2.cvtColor(vtd, cv2.COLOR_BGR2RGB))
                cv2.imshow('Qmap', cv2.cvtColor(vQmap, cv2.COLOR_BGR2RGB))
                cv2.waitKey(5000)

            metrics = {
                        'val_td': wandb.Image(vtd),
                        'val_Qmap': wandb.Image(vQmap),
                        }

            to_hist = {'Qmap': Qmap}
            for key, item in to_hist.items():
                metrics[f'val/{key}_hist'] = make_histogram(item, key)
            self.logger.log_metrics(metrics, self.global_step)

        return {'val_loss': torch.mean(batch_loss,axis=0),
                f'val_TD({self.hparams.n})_loss': torch.mean(td_loss,axis=0),
                'val_margin_loss': torch.mean(margin_loss,axis=0),}

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
        return get_dataloader(self.hparams, is_train=True)

    # online val dataloaders spoof a length of N batches, and do N episode rollouts
    def val_dataloader(self):
        return get_dataloader(self.hparams, is_train=False)


# offline training

def main(args):
    
    logger = False
    if args.log:
        logger = WandbLogger(id=args.id, save_dir=str(args.save_dir), project='dqn_test')
        #wandb.init(project='dqn_test')
    checkpoint_callback = ModelCheckpoint(args.save_dir, save_top_k=3) # figure out what's up with this
    # resume and add a couple arguments
    if args.restore_from is not None:
        if 'lbc' in args.restore_from:
            model = MapModel(args)
            model.restore_from_lbc(args.restore_from)
        elif 'dqn' in args.restore_from:
            model = MapModel.load_from_checkpoint(args.restore_from)
    else:
        model = MapModel(args)

    with open(args.save_dir / 'config.yml', 'w') as f:
        hparams_copy = copy.copy(vars(model.hparams))
        hparams_copy['dataset_dir'] = str(model.hparams.dataset_dir)
        hparams_copy['save_dir'] = str(model.hparams.save_dir)
        del hparams_copy['id']
        del hparams_copy['data_root']
        yaml.dump(hparams_copy, f, default_flow_style=False, sort_keys=False)
    shutil.copyfile(args.dataset_dir / 'config.yml', args.save_dir / 'data_config.yml')

    # offline trainer can use all gpus
    # when resuming, the network starts at epoch 36
    trainer = pl.Trainer(
        gpus=args.gpus, max_epochs=args.max_epochs,
        #resume_from_checkpoint=RESUME,
        logger=logger,
        checkpoint_callback=checkpoint_callback,
        distributed_backend='dp',)

    trainer.fit(model)

    if args.log:
        wandb.save(str(args.save_dir / '*.ckpt'))



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('-D', '--debug', action='store_true')
    parser.add_argument('-G', '--gpus', type=int, default=-1)
    parser.add_argument('--restore_from', type=str)
    parser.add_argument('--save_dir', type=Path)
    parser.add_argument('--id', type=str, default=datetime.now().strftime("%Y%m%d_%H%M%S")) 
    parser.add_argument('--data_root', type=Path, default='/data/aaronhua')
    parser.add_argument('--log', action='store_true')

    # Trainer args
    parser.add_argument('--dataset_dir', type=Path)
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
    parser.add_argument('--heatmap_radius', type=int, default=5)
    parser.add_argument('--expert_radius', type=int, default=2)
    parser.add_argument('--expert_margin', type=float, default=10.0)
    parser.add_argument('--temperature', type=float, default=10.0)
    parser.add_argument('--hack', action='store_true')
        
    args = parser.parse_args()

    if args.dataset_dir is None:
        args.dataset_dir = Path('/data/aaronhua/leaderboard/data/dqn/20210407_024101')

    suffix = f'debug/{args.id}' if args.debug else args.id
    save_root = args.data_root / f'leaderboard/training/dqn/offline/{suffix}'

    args.save_dir = save_root
    args.save_dir.mkdir(parents=True, exist_ok=True)

    main(args)
