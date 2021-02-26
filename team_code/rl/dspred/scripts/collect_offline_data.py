import os, sys, time
import yaml
import argparse 
from datetime import datetime
from pathlib import Path

sys.path.append('../..')
from common.utils import *

parser = argparse.ArgumentParser()

parser.add_argument('--restore_from', type=str, default=None)
parser.add_argument('--remote', action='store_true')
parser.add_argument('--data_root', type=str, default='/data')
parser.add_argument('-G', type=int, default=0)

# setup
parser.add_argument('--desc', type=str, default='no description')
parser.add_argument('--version', type=int, choices=[10,11], default=11) # 0.9.10.1 or 0.9.11
parser.add_argument('--split', type=str, default='devtest', choices=['debug', 'devtest', 'testing', 'training'])
parser.add_argument('--routenum', type=int) # optional
parser.add_argument('--scenarios', action='store_true') # leaderboard-triggered scnearios
parser.add_argument('--repetitions', type=int, default=1) # should we directly default to this in indexer?
parser.add_argument('--empty', action='store_true') # other agents present?
parser.add_argument('--save_images', action='store_true')
parser.add_argument('--verbose', action='store_true')
parser.add_argument('-d', '--debug', action='store_true')

args = parser.parse_args()

root = f'/home/aaronhua' if args.remote else f'/home/aaron/workspace/carla'

project_root = f'{root}/leaderboard-devkit'
config_path = f'{project_root}/team_code/rl/config/dspred.yml'
with open(config_path, 'r') as f:
    config = yaml.load(f, Loader=yaml.Loader)

# carla setup
carla_root = f'{root}/CARLA_0.9.{args.version}'
if args.version == 10:
    carla_root = f'{carla_root}.1'
carla_api = f'{carla_root}/PythonAPI/carla'
carla_egg = f'{carla_root}/PythonAPI/carla/dist/carla-0.9.{args.version}-py3.7-linux-x86_64.egg'

# leaderboard setup
routes = f'routes_{args.split}'
if args.routenum:
    routes = f'{routes}/route_{args.routenum:02d}'
routes = f'{routes}.xml'
scenarios = 'all_towns_traffic_scenarios_public.json' if args.scenarios \
        else 'no_traffic_scenarios.json'

# logging
date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
suffix = f'debug/{date_str}' if args.debug else f'{date_str}' 
save_root = f'{args.data_root}/leaderboard/data/rl/dspred/{suffix}'
mkdir_if_not_exists(f'{save_root}/logs')

# setup config
config['description'] = args.desc
config['remote'] = args.remote
config['project_root'] = project_root
config['carla_root'] = carla_root
config['save_root'] = save_root

# setup env config
econf = config['env']
econf['world_port'] = 2000
econf['trafficmanager_port'] = 8000
econf['trafficmanager_seed'] = 0
econf['routes'] = routes
econf['scenarios'] = scenarios
econf['repetitions'] = args.repetitions
econf['empty'] = args.empty
econf['hop_resolution'] = 2.0
econf['save_data'] = True


aconf = config['agent']
if args.debug:
    aconf['total_timesteps'] = 2000
    aconf['burn_timesteps'] = 250
    aconf['save_frequency'] = 500

total_timesteps = 2000 if args.debug else aconf['total_timesteps']
burn_timesteps = 250 if args.debug else 2000
save_frequency = 500 if args.debug else 5000

aconf['mode'] = 'train'
aconf['total_timesteps'] = total_timesteps
aconf['burn_timesteps'] = burn_timesteps
aconf['save_frequency'] = save_frequency
aconf['save_images'] = args.save_images

config_path = f'{save_root}/config.yml'
with open(config_path, 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.G)
os.environ["CONDA_ENV"] = 'dspred'
os.environ["PROJECT_ROOT"] = project_root
os.environ["SAVE_ROOT"] = save_root
os.environ["CARLA_EGG"] = carla_egg
os.environ["CARLA_API"] = carla_api
os.environ["RESTORE"] = "0"
os.environ["HAS_DISPLAY"] = str(int(not args.remote))

script = f'{project_root}/team_code/rl/dspred/collector.py'
cmd = f'bash {project_root}/team_code/scripts/run_script.sh {script} {config_path}'
os.system(cmd)
