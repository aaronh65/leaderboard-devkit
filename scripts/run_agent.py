#####################
# this script is used on my local machine
# you need to run CARLA before running this script

import os, sys, time
import yaml
import argparse
from datetime import datetime
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--split', type=str, default='devtest', choices=['devtest','testing','training','debug'])
parser.add_argument('--route', type=int, default=3)
parser.add_argument('--agent', type=str, default='lbc/image_agent', choices=['lbc/image_agent', 'lbc/auto_pilot', 'lbc/privileged_agent', 'rl/waypoint_agent', 'common/forward_agent'])
parser.add_argument('--repetitions', type=int, default=1)
parser.add_argument('--save_images', action='store_true')
parser.add_argument('--debug', action='store_true')
args = parser.parse_args()

# make base save path + log dir
project_root = "/home/aaron/workspace/carla/leaderboard-devkit"
date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
end_str = f'debug/{date_str}/{args.split}' if args.debug else f'{date_str}/{args.split}' 
save_root = f'/data/leaderboard/results/{args.agent}/{end_str}'
mkdir_if_not_exists(f'{save_root}/logs')

# route path
route_name = f'route_{args.route:02d}'
route_path = f'routes_{args.split}/{route_name}.xml'

# make image + performance plot dirs
if args.save_images:
    save_images_path = f'{save_root}/images/{route_name}'
    for rep_number in range(args.repetitions):
        mkdir_if_not_exists(f'{save_images_path}/repetition_{rep_number:02d}')
save_perf_path = f'{save_root}/plots/{route_name}'
mkdir_if_not_exists(save_perf_path)

# agent-specific configurations
config = {}
config['project_root'] = project_root
config['save_root'] = save_root
config['save_images'] = args.save_images
privileged = False
conda_env = 'lb'
if args.agent == 'common/forward_agent':
    pass
elif args.agent == 'lbc/auto_pilot':
    privileged = True
    config['save_data'] = False
elif args.agent == 'lbc/image_agent':
    conda_env = 'lblbc'
    config['weights_path'] = 'team_code/config/image_model.ckpt'
elif args.agent == 'lbc/privileged_agent':
    conda_env = 'lblbc'
    privileged = True
    config['weights_path'] = 'team_code/config/map_model.ckpt'
elif args.agent == 'rl/waypoint_agent':
    conda_env = 'lbrl'
    config['mode'] = 'train'
    config['world_port'] = 2000
    config['tm_port'] = 8000

config_path = f'{save_root}/config.yml'
with open(config_path, 'w') as f:
    yaml.dump(config, f, default_flow_style=False)

# environ variables
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PROJECT_ROOT"] = project_root
os.environ["SAVE_ROOT"] = save_root # call this save root
os.environ["CONDA_ENV"] = conda_env
os.environ["ROUTE_NAME"] = route_name
os.environ["WORLD_PORT"] = "2000"
os.environ["TM_PORT"] = "8000"
 
cmd = f'bash sh/run_agent.sh {args.agent} {route_path} {args.repetitions} {privileged}'
print(f'running {cmd}')
os.system(cmd)
