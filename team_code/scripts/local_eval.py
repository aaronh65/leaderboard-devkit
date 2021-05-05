#####################
# this script is used on my local machine
# you need to run CARLA before running this script

import os, sys, time, shutil
import yaml, argparse
from datetime import datetime
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument('-D', '--debug', action='store_true')
parser.add_argument('--agent', type=str, default='lbc/src/auto_pilot.py')
parser.add_argument('--split', type=str, default='devtest', 
        choices=['devtest','testing','training','debug'])
parser.add_argument('--routenum', type=int, default=3) # optional
parser.add_argument('--repetitions', type=int, default=1)

parser.add_argument('--id', type=str, default=datetime.now().strftime("%Y%m%d_%H%M%S"))
parser.add_argument('--save_data', action='store_true')
parser.add_argument('--save_debug', action='store_true')
parser.add_argument('--config_path', type=str)
args = parser.parse_args()

project_root = os.environ['PROJECT_ROOT']

# make save root + log dir
tokens = args.agent.split('.')[0].split('/')
appr, algo = tokens[0], tokens[-1]
prefix = '/data/aaronhua/leaderboard/data' if args.save_data else '/data/aaronhua/leaderboard/benchmark'
suffix = f'debug/{args.id}' if args.debug else args.id
save_root = Path(f'{prefix}/{appr}/{algo}/{suffix}')
save_root.mkdir(parents=True,exist_ok=True)
(save_root / 'plots').mkdir(exist_ok=True)
(save_root / 'logs').mkdir(exist_ok=True)

# agent-specific config
if args.config_path is None:
    config_path = f'{project_root}/team_code/{appr}/config/{algo}.yml'
else:
    config_path = f'{project_root}/team_code/{args.config_path}'
with open(config_path, 'r') as f:
    config = yaml.load(f, Loader=yaml.Loader)

config['project_root'] = project_root
config['save_root'] = str(save_root)
config['save_data'] = args.save_data
config['save_debug'] = args.save_debug
config['split'] = args.split
config['repetitions'] = args.repetitions

track = 'SENSORS'
privileged = algo in ['auto_pilot', 'privileged']

config_path = f'{save_root}/config.yml'
print(config_path)
with open(config_path, 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
if 'weights_path' in config.keys() or ('agent' in config.keys() and 'weights_path' in config['agent'].keys()):
        if 'weights_path' in config.keys():
            wpath = Path(config['weights_path']).parent
        else:
            wpath = Path(config['agent']['weights_path']).parent
        shutil.copyfile(wpath / 'config.yml', save_root / 'train_config.yml')
        shutil.copyfile(wpath / 'train_data_config.yml', save_root / 'train_data_config.yml')
        shutil.copyfile(wpath / 'val_data_config.yml', save_root / 'val_data_config.yml')

# environ variables
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["SAVE_ROOT"] = str(save_root)
os.environ["TRACK"] = track
os.environ["WORLD_PORT"] = "2000"
os.environ["TM_PORT"] = "8000"
os.environ["AGENT"] = args.agent
os.environ["SPLIT"] = args.split
os.environ["ROUTE_NAME"] = f'route_{args.routenum:02d}'
os.environ["REPETITIONS"] = str(args.repetitions)
os.environ["PRIVILEGED"] = str(int(privileged))
 
cmd = f'bash run_leaderboard.sh'
print(f'running {cmd} on {args.split} routes for {args.repetitions} repetitions')
os.system(cmd)
