import yaml
from pathlib import Path
import numpy as np
import argparse
from dqn.src.offline.split_dataset import get_dataloader
#import matplotlib as mpl
#mpl.use('TkAgg')
#import matplotlib.pyplot as plt
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
parser.add_argument('--throttle_mode', type=str, default='speed')
parser.add_argument('--max_speed', type=int, default=10)
parser.add_argument('--hard_prop', type=float, default=0.5)
parser.add_argument('--max_prop_epoch', type=float, default=0.5)
parser.add_argument('--max_epochs', type=int, default=10)

args = parser.parse_args()
loader = get_dataloader(args, args.dataset_dir, is_train=True)


meas = loader.dataset.measurements
infractions = meas['infraction'].tolist()
infraction_names = [i for i in set(infractions) if i != 'none']
counts_d = {}
for infraction in infractions:
    if infraction == 'none':
        continue
    if infraction not in counts_d.keys():
        counts_d[infraction] = 0
    counts_d[infraction] += 1
save_root = Path('infraction_counts')
save_root.mkdir(exist_ok=True)
dataset_name = args.dataset_dir.stem

metrics = {
    'num_infractions': loader.dataset.infraction_count,
    'num_hard_frames': len(loader.dataset.hard_indices),
    }
metrics.update(counts_d)
with open(str(save_root / f'{dataset_name}.yml'), 'w') as f:
    yaml.dump(metrics, f, default_flow_style=False)
#counts = [counts_d[i] for i in infraction_names]
#infraction_names = [i.split('.')[1] for i in infraction_names]
#plt.bar(1.5*np.arange(len(counts)), counts)
#plt.xticks(1.5*np.arange(len(counts)), infraction_names)
#plt.show()
