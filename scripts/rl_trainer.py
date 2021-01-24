import os, sys, time
import yaml
import argparse 
from datetime import datetime
from utils import *

parser = argparse.ArgumentParser()

parser.add_argument('--version', type=int, choices=[10,11], default=11)

# indexing + setup
parser.add_argument('--split', type=str, default='training', choices=['debug', 'devtest', 'testing', 'training'])
parser.add_argument('--routenum', type=int)
parser.add_argument('--scenarios', action='store_true')
parser.add_argument('--repetitions', type=int, default=1)
parser.add_argument('--empty', action='store_true')

# logging
parser.add_argument('--restore_from', type=str)
parser.add_argument('--save_images', action='store_true')
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--debug', action='store_true')
args = parser.parse_args()

# set carla version variables
carla_root = f'/home/aaron/workspace/carla/CARLA_0.9.{args.version}'
if args.version == 10:
    carla_root = f'{carla_root}.1'
carla_api = f'{carla_root}/PythonAPI/carla'
carla_egg = f'{carla_root}/PythonAPI/carla/dist/carla-0.9.{args.version}-py3.7-linux-x86_64.egg'

# save path for images/logs/videos/plots
restore = args.restore_from is not None
if restore:
    config_path = f'{args.restore_from}/config.yml'
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)
    project_root = config['sac_config']['project_root']
    save_root = config['sac_config']['save_root']
else:
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

    if args.debug:
        total_timesteps = 1000
        burn_timesteps = 400
    else:
        total_timesteps = 500000
        burn_timesteps = 2000

    project_root = '/home/aaron/workspace/carla/leaderboard-devkit'
    env_config = {
            'save_root': save_root,
            'project_root': project_root,
            'save_images': args.save_images,
            'world_port': 2000,
            'trafficmanager_port': 8000,
            'trafficmanager_seed': 0,
            'routes': routes,
            'scenarios': scenarios,
            'repetitions': args.repetitions,
            'empty': args.empty,
            }

    sac_config = {
            'save_root': save_root,
            'project_root': project_root,
            'save_images': args.save_images,
            'mode': 'train',
            'total_timesteps': total_timesteps,
            'burn_timesteps': burn_timesteps,
            'train_frequency': 1,
            'gradient_steps': 1,
            'target_update_interval': 1,
            'save_frequency': 1000,
            'log_frequency': 1000,
            'verbose': args.verbose,
            }

    config = {'env_config': env_config, 'sac_config': sac_config}
    config_path = f'{save_root}/config.yml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

os.environ["CONDA_ENV"] = 'lbrl'
os.environ["PROJECT_ROOT"] = project_root
os.environ["SAVE_ROOT"] = save_root
os.environ["CARLA_EGG"] = carla_egg
os.environ["CARLA_API"] = carla_api
os.environ["RESTORE"] = "1" if restore is not None else "0"

cmd = f'bash sh/rl_trainer.sh {config_path}'
os.system(cmd)
