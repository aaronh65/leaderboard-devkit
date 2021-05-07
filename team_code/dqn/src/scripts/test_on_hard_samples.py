import argparse
import cv2
from pathlib import Path
from torch.utils.data import DataLoader
from dqn.src.offline.split_dataset import SplitCarlaDataset, get_dataloader
from dqn.src.agents.map_model import MapModel
from dqn.src.agents.heatmap import ToHeatmap


model1_path = '/data/aaronhua/leaderboard/training/dqn/offline/20210505_012133/last.ckpt'
model2_path = '/data/aaronhua/leaderboard/training/dqn/offline/20210506_003502/last.ckpt'

to_heatmap = ToHeatmap(5)
parser = argparse.ArgumentParser()
#parser.add_argument('--dataset_dir', type=Path,
#        default='/data/aaronhua/leaderboard/data/lbc/autopilot/autopilot_devtest_toy')
parser.add_argument('--dataset_dir', type=Path, 
        default='/data/aaronhua/leaderboard/data/lbc/autopilot/autopilot_devtest')
#parser.add_argument('--weights_path', type=str,
#        default='/data/aaronhua/leaderboard/training/lbc/20210405_225046/epoch=22.ckpt')
parser.add_argument('--n', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--save_visuals', action='store_true')
parser.add_argument('--hard_prop', type=float, default=1.0)
parser.add_argument('--max_epochs', type=int, default=10)
parser.add_argument('--max_prop_epoch', type=int, default=0)
args = parser.parse_args()
dataset_dir = Path(args.dataset_dir) / 'data'


loader = get_dataloader(args,args.dataset_dir,is_train=True)
model1 = MapModel.load_from_checkpoint(model1_path)
model1.logger = None
model1.cuda()

model2 = MapModel.load_from_checkpoint(model2_path)
model2.logger = None
model2.cuda()

for batch_nb, batch in enumerate(loader):
    state, action, reward, next_state, done, info = batch
    state[0] = state[0].cuda()
    state[1] = state[1].cuda()
    next_state[0] = next_state[0].cuda()
    next_state[1] = next_state[1].cuda()
    action = action.cuda()
    reward = reward.cuda()
    done = done.cuda()
    for key, item in info.items():
        info[key] = item.cuda()
    batch = state, action, reward, next_state, done, info
    model1.validation_step(batch, batch_nb, show=True, no_waitkey=1)
    model2.validation_step(batch, batch_nb, show=True, no_waitkey=2)
    cv2.waitKey(0)


#episodes = list()

#routes = sorted(list(dataset_dir.glob('*')))
#for route in routes:
#    episodes.extend(sorted(route.glob('*')))

#dataset = SplitCarlaDataset(args, episodes, is_train=True)
#for batch_nb, batch in enumerate(dataset):
#    (topdown target), points_expert, reward, (ntopdown, ntarget), done, info = batch



