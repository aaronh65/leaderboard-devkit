import os, argparse, copy, shutil, yaml
from pathlib import Path
from datetime import datetime

from PIL import Image, ImageDraw
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

# takes (N,3,H,W) topdown and (N,4,H,W) logits
# averages t=0.5s,1.0s logitss and overlays it on topdown
@torch.no_grad()
def fuse_logits(topdown, logits, alpha=0.5):

    #logits_mean = torch.mean(logits[:,0:2,:,:], dim=1, keepdim=True) # N,1,H,W
    logits_norm = spatial_norm(logits).cpu().numpy()*255 #N,T,H,W
    logits_norm = np.expand_dims(logits_norm,axis=-1).astype(np.uint8) 
    #print(logits_norm.squeeze()[0][0])
    #print(logits_norm.squeeze()[0][1])
    logits_show = np.tile(logits_norm, (1,1,1,1,3)) #N,T,H,W,3
    #logits_show = np.repeat(logits_norm, repeats=3, axis=-1) #N,T,H,W,3
    #logits_flat = logits_mean.view(logits_mean.shape[:-2] + (-1,)) # N,1,H*W
    #logits_prob = F.softmax(logits_flat/temperature, dim=-1) # to prob
    #logits_norm = logits_prob / torch.max(logits_prob, dim=-1, keepdim=True)[0] # to [0,1]
    #logits_show = (logits_norm * 256).view_as(logits_mean).cpu().numpy().astype(np.uint8) # N,1,H,W
    #logits_show = np.repeat(logits_norm, repeats=3, axis=1).transpose((0,2,3,1)) # N,H,W,3
    fused = np.array(COLOR[topdown.argmax(1).cpu()]).astype(np.uint8) # N,H,W,3
    for i in range(fused.shape[0]):
        logits_mask = cv2.addWeighted(logits_show[i][0], 1, logits_show[i][1], 1, 0)
        fused[i] = cv2.addWeighted(logits_mask, alpha, fused[i], 1, 0)
    fused = fused.astype(np.uint8) # (N,H,W,3)
    return fused

# visualize each timestep's heatmap?
@torch.no_grad()
def visualize(batch, Qmap, tmap, nQmap, ntmap, naction, meta, r=2,title='topdown'):
    #print(title)

    text_color = (255,255,255)
    aim_color = (60,179,113) # dark green
    student_color = (65,105,225) # dark blue
    expert_color = (178,34,34) # dark red
    route_colors = [(255,255,255), (112,128,144), (47,79,79), (47,79,79)] 

    state, action, reward, next_state, done, info = batch
    batch_loss, td_loss, margin_loss = meta['batch_loss'], meta['td_loss'], meta['margin_loss']
    hparams = meta['hparams']
    Q, nQ = meta['Q'], meta['nQ']
    discount = info['discount'].cpu()
    n = hparams.n

    topdown, target = state
    fused = fuse_logits(topdown, Qmap)

    ntopdown, ntarget = next_state
    nfused = fuse_logits(ntopdown, nQmap)

    images = list()
    for i in range(min(action.shape[0], 32)):

        # current state
        _topdown = fused[i]
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
        _draw.text((5, 30), f'done = {bool(done[i])}', text_color)
        _metadata = decode_str(info['metadata'][i])
        _draw.text((5, 40), f'meta = {_metadata}', text_color)

        # next state
        _ntopdown = nfused[i]
        _ntopdown[ntmap[i][0].cpu() > 0.1] = 255
        _ntopdown = Image.fromarray(_ntopdown)
        _ndraw = ImageDraw.Draw(_ntopdown)
        _naction = naction[i].cpu().numpy().astype(np.uint8) # (4,2)
        for x, y in _naction:
            _ndraw.ellipse((x-r, y-r, x+r, y+r), student_color)
        x, y = np.mean(_naction[0:2], axis=0)
        _ndraw.ellipse((x-r, y-r, x+r, y+r), aim_color)

        _ndraw.text((5, 10), f'nQ = {nQ[i].item():.2f}', text_color)
        _ndraw.text((5, 20), f'discount = {discount[i].item():.2f}', text_color)
        _td_loss = f'{td_loss[i].item():.2f}'
        _margin_loss = f'{margin_loss[i].item():.2f}'
        _batch_loss = f'{batch_loss[i].item():.2f}'
        _ndraw.text((5, 30), f'batch_loss = {_td_loss} + {_margin_loss} = {_batch_loss}', text_color)

        _combined = np.hstack((np.array(_topdown), np.array(_ntopdown)))
        #if HAS_DISPLAY:
        #    #cv2.imshow(f'{title}_{i}', cv2.cvtColor(_combined, cv2.COLOR_BGR2RGB))
        #    pass
        images.append((batch_loss[i].item(), _combined))
        #images.append((td_loss[i].item(), torch.ByteTensor(_combined)))

    #if HAS_DISPLAY:
    #    cv2.waitKey(5000)
    #    pass

    images.sort(key=lambda x: x[0], reverse=True)
    result = [torch.ByteTensor(x[1].transpose(2,0,1)) for x in images]
    result = torchvision.utils.make_grid(result, nrow=4)
    result = result.numpy().transpose(1,2,0)
    result = wandb.Image(result)
    if HAS_DISPLAY:
        images = [x[1] for x in images]
        combined = np.hstack(images)
        cv2.imshow(title, combined)
        cv2.waitKey(5000)

    return images, result

@torch.no_grad()
def make_histogram(tensor, in_type, b_i=0, c_i=0):
    data = tensor.detach().cpu().numpy()[b_i][c_i].flatten()
    if len(data) > 10000:
        data = data[-10000:]
    data = [[item] for item in data] 
    table = wandb.Table(data=data, columns=[in_type])
    hist = wandb.plot.histogram(table, in_type, title=f'{in_type}/b{b_i}/c{c_i} distribution')
    return hist

# just needs to know if it's rolling out or nah
class MapModel(pl.LightningModule):
    def __init__(self, hparams=None):
        super().__init__()

        if hparams is not None:
            self.hparams = hparams
            self.to_heatmap = ToHeatmap(hparams.heatmap_radius)
            self.register_buffer('temperature', torch.Tensor([self.hparams.temperature]))
        else:
            self.to_heatmap = ToHeatmap(5)
            self.register_buffer('temperature', torch.Tensor([10]))

        self.net = SegmentationModel(10, 4, batch_norm=True, hack=True)
        self.controller = RawController(4)
        self.td_criterion = torch.nn.MSELoss(reduction='none') # weights? prioritized replay?
        self.expert_heatmap = ToTemporalHeatmap(5)
        self.margin_criterion = torch.nn.MSELoss(reduction='none')
        #self.Q_conv = torch.nn.Conv2d(in_channels=4,out_channels=4,kernel_size=1,groups=4)
        self.margin_weight = 100
        #self.Q_mult = torch.nn.Parameter(torch.rand(1))
        #self.Q_bias = torch.nn.Parameter(torch.rand(1))

    #def on_train_start(self):
    #    #print('on_train_start')
    #    #print(self.logger is not None)
    #    #if self.logger is not None:
    #    #    self.logger.watch(self)
    #    pass

    def restore_from_lbc(self, weights_path):
        from lbc.carla_project.src.map_model import MapModel as LBCModel

        print('restoring from LBC')
        lbc_model = LBCModel.load_from_checkpoint(weights_path)
        self.net.load_state_dict(lbc_model.net.state_dict())
        self.temperature = lbc_model.temperature
        self.to_heatmap = lbc_model.to_heatmap

    def forward(self, topdown, target, debug=True):
        target_heatmap = self.to_heatmap(target, topdown)[:, None]
        input = torch.cat((topdown, target_heatmap), 1)
        # points (N,T,2), logits/weights (N,T,H,W)
        points, logits, weights = self.net(input, temperature=self.temperature.item())
        #Q_unscaled = weights * 2 - 1 #  (0,1) -> (-1,1)
        #$Qmap = Q_unscaled * self.Q_mult + self.Q_bias # (N,T,H,W)

        #$Qmap = logits
        #Qmap = self.Q_conv(weights) # weights are [0,1]
        #Qmap = self.Q_conv(logits) # weights are [0,1]
        Qmap = logits
        return points, logits, weights, target_heatmap, Qmap

        # two modes: sample, argmax?
    def get_dqn_actions(self, Qmap, explore=False):
        h,w = Qmap.shape[-2:]
        Qmap_flat = Qmap.view(Qmap.shape[:-2] + (-1,)) # (N, 4, H*W)
        Q_all, action_flat = torch.max(Qmap_flat, -1, keepdim=True) # (N, 4, 1)

        action = torch.cat((
            action_flat % w,
            action_flat // w),
            axis=2) # (N, C, 2)
        
        return action, Q_all # (N,4,2), (N,4,1)

    def get_Q_values(self, Qmap, action):
        x, y = action[...,0], action[...,1] # Nx4
        action_flat = torch.unsqueeze(y*256 + x, dim=2).long() # (N,4, 1)
        action_flat = torch.clamp(action_flat, 0, 256*256-1)
        Qmap_flat = Qmap.view(Qmap.shape[:-2] + (-1,)) # (N, 4, H*W)
        Q_all = Qmap_flat.gather(dim=-1, index=action_flat) # (N, 4, 1)
        #print(Q_all.shape)
        return Q_all # (N,4,1)

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
        #Q_expert = torch.mean(Q_expert_all.squeeze(), axis=-1, keepdim=True) #N,1

        margin_map = self.expert_heatmap(points_expert, Qmap) #[0,1] tall at expert points
        margin_map = (1-margin_map) #[0, 8] low at expert points
        margin = Qmap + margin_map  - Q_expert_all.unsqueeze(-1)
        #cv2.imshow('margin', spatial_norm(margin).detach().cpu().numpy()[0][0])
        margin_loss = torch.mean(margin, dim=(1,2,3)) #N,
        margin_loss = self.hparams.lambda_margin * margin_loss.unsqueeze(-1)

        batch_loss = td_loss + margin_loss #N,1
        loss = torch.mean(batch_loss, dim=0) #1,

        if batch_nb % 50 == 0:
            meta = {
                    'Q': Q, 'nQ': nQ, 'hparams': self.hparams,
                    'batch_loss': batch_loss,
                    'td_loss': td_loss, 
                    'margin_loss': margin_loss,
                    'metadata': info['metadata'],
                    }

            images, result = visualize(batch, logits, tmap, nlogits, ntmap, naction, meta, title='logits')


        if self.logger != None:
            metrics = {
                        f'train/TD({self.hparams.n}) loss': td_loss.mean().item(),
                        'train/margin_loss': margin_loss.mean().item(),
                        'train/batch_loss': batch_loss.mean().item(),
                        }

            if batch_nb % 50 == 0:
                #self.logger.log_metrics({'train_viz': result}, self.global_step)
                # TODO: handle debug images
                #if self.config.save_debug:
                #    img = cv2.cvtColor(np.array(self.env.hero_agent.debug_img), cv2.COLOR_RGB2BGR)
                #    metrics['debug_image'] = wandb.Image(img)

                                #images, result = visualize(batch, weights, tmap, nweights, ntmap, naction, meta, title='weights')
                #images, result = visualize(batch, Qmap, tmap, nQmap, ntmap, naction, meta, title='Qmap')

                #images, result = visualize(batch, weights, tmap, nweights, ntmap, naction, meta)
                #images, result = visualize(batch, Qmap, tmap, nQmap, ntmap, naction, meta)
                            #cv2.waitKey(5000)
                            
                #metrics['Q_w'] = self.Q_conv.weight.data.squeeze().cpu().numpy()
                #metrics['Q_b'] = self.Q_conv.bias.data.squeeze().cpu().numpy()
                metrics['train_image'] = result

                to_hist = {'logits': logits, 'weights': weights, 'Qmap': Qmap}
                for key, item in to_hist.items():
                    metrics[f'train/{key}_hist'] = make_histogram(item, key)
            self.logger.log_metrics(metrics, self.global_step)

        return {'loss': loss}

    # make this a validation episode rollout?
    def validation_step(self, batch, batch_nb):
        metrics ={}

        state, action, reward, next_state, done, info = batch
        topdown, target = state
        points, logits, weights, tmap, Qmap = self.forward(topdown, target, debug=True)
        Q_all = spatial_select(Qmap, action).squeeze() #NxT
        Q = torch.mean(Q_all, axis=-1, keepdim=True)

        ntopdown, ntarget = next_state
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
        Q_expert = torch.mean(Q_expert_all.squeeze(), axis=-1, keepdim=True) #N,1


        #if True:
        #    hmaps = np.hstack(margin_map[0].cpu().numpy())
        #    print(hmaps.shape)
        #    cv2.imshow('hmaps',hmaps)
        #    cv2.waitKey(0)
        margin_map = self.expert_heatmap(points_expert, Qmap) #[0,1] tall at expert points
        margin_map = (1-margin_map) #[0, 8] low at expert points
        margin = Qmap + margin_map  - Q_expert_all.unsqueeze(-1)
        #cv2.imshow('margin', margin.detach().cpu().numpy()[0][0])
        margin_loss = torch.mean(margin, dim=(1,2,3)) #N,
        margin_loss = self.hparams.lambda_margin * margin_loss.unsqueeze(-1)


        batch_loss = td_loss + margin_loss
        val_loss = torch.mean(batch_loss, axis=0)
        
                
        meta = {
                'Q': Q, 'nQ': nQ, 'hparams': self.hparams, 
                'batch_loss': batch_loss,
                'td_loss': td_loss,
                'margin_loss': margin_loss,
                'metadata': info['metadata']
                }

        images, result = visualize(batch, logits, tmap, nlogits, ntmap, naction, meta, title='logits')

        if self.logger != None:
            #print(result)
            #print(result.shape)
            #self.logger.log_metrics({'val_viz': result}, self.global_step)

            if HAS_DISPLAY:
                images = [x[1] for x in images]
                combined = np.hstack(images)
                cv2.imshow('logits', combined)
                #cv2.waitKey(0)

            metrics = {
                        #'Q_w' = self.Q_conv.weight.data.squeeze().cpu().numpy(),
                        #'Q_b' = self.Q_conv.bias.data.squeeze().cpu().numpy(),
                        'val_image': result,
                        f'val/TD({self.hparams.n}) loss': td_loss.mean().item(),
                        'val/margin_loss': margin_loss.mean().item(),
                        'val/batch_loss': batch_loss.mean().item(),
                        }

            #to_hist = {'logits': logits, 'weights': weights, 'Qmap': Qmap}
            #for key, item in to_hist.items():
            #    metrics[f'{key}_hist'] = make_histogram(item, key)
            #self.logger.log_metrics(metrics, self.global_step)

            self.logger.log_metrics(metrics, self.global_step)
            #self.logger.log_metrics(loss_metrics, self.global_step)
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
    checkpoint_callback = ModelCheckpoint(args.save_dir, save_top_k=1) # figure out what's up with this
    # resume and add a couple arguments
    if args.restore_from is not None:
        if 'lbc' in args.restore_from:
            model = MapModel(args)
            model.restore_from_lbc(args.restore_from)
        elif 'dqn' in args.restore_from:
            model = MapModel.load_from_checkpoint(args.restore_from)
            pass
    else:
        model = MapModel(args)




    #model = MapModel.load_from_checkpoint(RESUME)
    #model.hparams.max_epochs = args.max_epochs
    #model.hparams.dataset_dir = args.dataset_dir
    #model.hparams.batch_size = args.batch_size
    #model.hparams.save_dir = args.save_dir
    #model.hparams.n = args.n
    #model.hparams.gamma = args.gamma
    #model.hparams.num_workers = args.num_workers
    #model.hparams.no_margin = args.no_margin
    #model.hparams.no_td = args.no_td
    #model.hparams.data_mode = 'offline'

    with open(args.save_dir / 'config.yml', 'w') as f:
        hparams_copy = copy.copy(vars(model.hparams))
        hparams_copy['dataset_dir'] = str(model.hparams.dataset_dir)
        hparams_copy['save_dir'] = str(model.hparams.save_dir)
        del hparams_copy['id']
        del hparams_copy['data_root']
        yaml.dump(hparams_copy, f, default_flow_style=False, sort_keys=False)

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
    parser.add_argument('--temperature', type=float, default=10.0)
    parser.add_argument('--hack', action='store_true', default=False) # what is this again?
        
    args = parser.parse_args()

    if args.dataset_dir is None:
        args.dataset_dir = '/data/aaronhua/leaderboard/data/dqn/20210407_024101'

    suffix = f'debug/{args.id}' if args.debug else args.id
    save_root = args.data_root / f'leaderboard/training/dqn/offline/{suffix}'

    args.save_dir = save_root
    args.save_dir.mkdir(parents=True, exist_ok=True)

    main(args)
