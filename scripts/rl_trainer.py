import os, sys, time
import yaml
import argparse 
from datetime import datetime
from utils import *

parser = argparse.ArgumentParser()

# route indexing
parser.add_argument('--split', type=str, default='training', choices=['debug', 'devtest', 'testing', 'training'])
parser.add_argument('--routenum', type=int)
parser.add_argument('--scenarios', action='store_true')
parser.add_argument('--repetitions', type=int, default=1)

# other actors?
parser.add_argument('--empty', action='store_true')

# logging
parser.add_argument('--save_images', action='store_true')
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--debug', action='store_true')
args = parser.parse_args()

# save path for images/logs/videos/plots
project_root = "/home/aaron/workspace/carla/leaderboard-devkit"
date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
suffix = f'debug/{date_str}' if args.debug else f'{date_str}' 
save_root = f'/data/leaderboard/results/rl/waypoint_agent/{suffix}'
if args.save_images:
    mkdir_if_not_exists(f'{save_root}/images')
mkdir_if_not_exists(f'{save_root}/weights')

# route indexer information
routes = f'routes_{args.split}'
if args.routenum:
    routes = f'{routes}/route_{args.routenum:02d}'
routes = f'{routes}.xml'
scenarios = 'all_towns_traffic_scenarios_public.json' if args.scenarios \
        else 'no_traffic_scenarios.json'

env_config = {
        'world_port': 2000,
        'trafficmanager_port': 8000,
        'trafficmanager_seed': 0,
        'routes': routes,
        'scenarios': scenarios,
        'repetitions': args.repetitions,
        'empty': args.empty,
        'save_images': args.save_images,
        }

sac_config = {
        'save_root': save_root,
        'mode': 'train',
        'total_timesteps': 500000,
        'burn_timesteps': 2000,
        #'total_timesteps': 1000,
        #'burn_timesteps': 100,
        'train_frequency': 1,
        'gradient_steps': 1,
        'target_update_interval': 1,
        'save_frequency': 1000,
        'log_frequency': 1000,
        'verbose': args.verbose,
        'save_images': args.save_images,
        }

config = {'env_config': env_config, 'sac_config': sac_config}
config_path = f'{save_root}/config.yml'
with open(config_path, 'w') as f:
    yaml.dump(config, f, default_flow_style=False)

os.environ["PROJECT_ROOT"] = project_root
os.environ["SAVE_ROOT"] = save_root
os.environ["CONDA_ENV"] = 'lbrl'

cmd = f'bash sh/rl_trainer.sh {config_path}'
os.system(cmd)
