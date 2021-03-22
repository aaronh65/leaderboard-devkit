import os, sys, time
sys.path.append('../..')
import yaml
import argparse 
from datetime import datetime
from common.utils import *


# copy auto pilot code
# add options to make the environment empty
parser = argparse.ArgumentParser()

parser.add_argument('--version', type=int, choices=[10,11], default=11)
parser.add_argument('--num_steps', type=int, default=100)
parser.add_argument('--scenarios', action='store_true')
parser.add_argument('--empty', action='store_true')
parser.add_argument('--repetitions', type=int, default=1)
parser.add_argument('--split', type=str, default='training', 
        choices=['debug','devtest','testing','training'])
parser.add_argument('--routenum', type=int)

parser.add_argument('-d', '--desc', type=str, default='no description')
parser.add_argument('--debug', action='store_true')

args = parser.parse_args()

carla_root = f'/home/aaron/workspace/carla/CARLA_0.9.{args.version}'
if args.version == 10:
    carla_root = f'{carla_root}.1'
carla_api = f'{carla_root}/PythonAPI/carla'
carla_egg = f'{carla_root}/PythonAPI/carla/dist/carla-0.9.{args.version}-py3.7-linux-x86_64.egg'

routes = f'routes_{args.split}'
if args.routenum:
    routes = f'{routes}/route_{args.routenum:02d}'
routes = f'{routes}.xml'
scenarios = 'all_towns_traffic_scenarios_public.json' if args.scenarios \
        else 'no_traffic_scenarios.json'

date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
suffix = f'debug/{date_str}' if args.debug else f'{date_str}' 
save_root = f'/data/leaderboard/data/rl/semantic_enc/{suffix}'
mkdir_if_not_exists(save_root)

project_root = '/home/aaron/workspace/carla/leaderboard-devkit'

env_config = {
        'carla_version': carla_root.split('/')[-1],
        'world_port': 2000,
        'trafficmanager_port': 8000,
        'trafficmanager_seed': 0,
        'routes': routes,
        'scenarios': scenarios,
        'repetitions': args.repetitions,
        'empty': args.empty,
        'hop_resolution': 2,
        'num_steps': args.num_steps,
        }

agent_config = {
        'save_data': True,
        }

config = {
        'description': args.desc, 
        'project_root': project_root,
        'save_root': save_root,
        'env': env_config, 
        'agent': agent_config
        }
config_path = f'{save_root}/config.yml'
with open(config_path, 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

os.environ["CONDA_ENV"] = 'lblbc'
#os.environ["CONDA_ENV"] = 'lbrl'
os.environ["AGENT"] = 'lbc/auto_pilot'
os.environ["PROJECT_ROOT"] = project_root
os.environ["SAVE_ROOT"] = save_root
os.environ["CARLA_EGG"] = carla_egg
os.environ["CARLA_API"] = carla_api

cmd = f'bash run_collect_semantic_data.sh {config_path}'
print(cmd)
os.system(cmd)

