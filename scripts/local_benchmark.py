#####################
# this script is used on my local machine
# you need to run CARLA before running this script

import os, sys, time
import yaml
import argparse
from datetime import datetime
from utils import mkdir_if_not_exists

parser = argparse.ArgumentParser()
parser.add_argument('--version', type=int, choices=[10,11], default=11)
parser.add_argument('--split', type=str, default='devtest', choices=['devtest','testing','training','debug'])
parser.add_argument('--route', type=int, default=3)
parser.add_argument('--agent', type=str, default='lbc/image_agent', choices=['lbc/image_agent', 'lbc/auto_pilot', 'lbc/privileged_agent', 'rl/waypoint_agent', 'common/forward_agent'])
parser.add_argument('--repetitions', type=int, default=1)
parser.add_argument('--save_images', action='store_true')
parser.add_argument('--debug', action='store_true')
args = parser.parse_args()

# set carla version variables
carla_root = f'/home/aaron/workspace/carla/CARLA_0.9.{args.version}'
if args.version == 10:
    carla_root = f'{carla_root}.1'
carla_api = f'{carla_root}/PythonAPI/carla'
carla_egg = f'{carla_root}/PythonAPI/carla/dist/carla-0.9.{args.version}-py3.7-linux-x86_64.egg'

# make save root + log dir
date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
suffix = f'debug/{date_str}' if args.debug else f'{date_str}' 
save_root = f'/data/leaderboard/results/{args.agent}/{suffix}'
mkdir_if_not_exists(f'{save_root}/logs')

# route information
route_name = f'route_{args.route:02d}'
route_path = f'routes_{args.split}/{route_name}.xml'

# images and plot saving
if args.save_images:
    images_path = f'{save_root}/images/{route_name}'
    for rep_number in range(args.repetitions):
        mkdir_if_not_exists(f'{images_path}/repetition_{rep_number:02d}')
plots_path = f'{save_root}/plots/{route_name}'
mkdir_if_not_exists(plots_path)

project_root = "/home/aaron/workspace/carla/leaderboard-devkit"

# agent-specific configurations
config = {}
config['project_root'] = project_root
config['save_root'] = save_root
config['save_images'] = args.save_images
config['split'] = args.split
privileged = False
conda_env = 'lb'
if args.agent == 'common/forward_agent':
    pass
elif args.agent == 'lbc/auto_pilot':
    conda_env = 'lblbc'
    privileged = True
    config['save_data'] = False
elif args.agent == 'lbc/image_agent':
    conda_env = 'lblbc'
    config['weights_path'] = 'team_code/config/image_model.ckpt'
elif args.agent == 'lbc/privileged_agent':
    conda_env = 'lblbc'
    privileged = True
    config['weights_path'] = 'team_code/config/map_model.ckpt'

config_path = f'{save_root}/config.yml'
with open(config_path, 'w') as f:
    yaml.dump(config, f, default_flow_style=False)

# environ variables
os.environ["CONDA_ENV"] = conda_env
os.environ["PROJECT_ROOT"] = project_root
os.environ["SAVE_ROOT"] = save_root
os.environ["CARLA_EGG"] = carla_egg
os.environ["CARLA_API"] = carla_api
os.environ["HAS_DISPLAY"] = "1"

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WORLD_PORT"] = "2000"
os.environ["TM_PORT"] = "8000"
os.environ["AGENT"] = args.agent
os.environ["SPLIT"] = args.split
os.environ["ROUTE_NAME"] = route_name
os.environ["REPETITIONS"] = str(args.repetitions)
os.environ["PRIVILEGED"] = "1" if privileged else "0"
 
cmd = f'bash sh/run_agent.sh'
print(f'running {cmd} on {args.split}/{route_name} for {args.repetitions} repetitions')
os.system(cmd)
