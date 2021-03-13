#####################
# this script is used on my local machine
# you need to run CARLA before running this script

import os, sys, time
import yaml, argparse
from datetime import datetime
from pathlib import Path

parser = argparse.ArgumentParser()

parser.add_argument('-D', '--debug', action='store_true')
parser.add_argument('--agent', type=str, default='lbc/privileged', 
        choices=['lbc/image', 'lbc/autopilot', 'lbc/privileged', 'common/forward', 'lbc/xodrmap', 'rl/dspred'])
parser.add_argument('--split', type=str, default='devtest', 
        choices=['devtest','testing','training','debug'])
parser.add_argument('--routenum', type=int, default=3) # optional
parser.add_argument('--repetitions', type=int, default=1)

parser.add_argument('--id', type=str, default=datetime.now().strftime("%Y%m%d_%H%M%S"))
parser.add_argument('--save_debug', action='store_true')
parser.add_argument('--save_data', action='store_true')
args = parser.parse_args()


# make save root + log dir
project_root = os.environ['PROJECT_ROOT']
suffix = f'debug/{args.id}' if args.debug else args.id
save_root = Path(f'/data/leaderboard/results/{args.agent}/{suffix}')
save_root.mkdir(parents=True,exist_ok=True)
(save_root / 'plots').mkdir(exist_ok=True)
(save_root / 'logs').mkdir(exist_ok=True)


# agent-specific configurations
appr, algo = args.agent.split('/')
config_path = f'{project_root}/team_code/{appr}/config/{algo}.yml'
with open(config_path, 'r') as f:
    config = yaml.load(f, Loader=yaml.Loader)

config['project_root'] = project_root
config['save_root'] = str(save_root)
config['save_debug'] = args.save_debug
config['save_data'] = args.save_data
config['split'] = args.split

track = 'SENSORS' if algo is not 'xodrmap' else 'MAP'
privileged = algo in ['auto_pilot', 'privileged']

config_path = f'{save_root}/config.yml'
with open(config_path, 'w') as f:
    yaml.dump(config, f, default_flow_style=False)

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
