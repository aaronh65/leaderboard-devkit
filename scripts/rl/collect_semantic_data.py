import os, sys, time
sys.path.append('..')
import yaml
import argparse 
from datetime import datetime
from utils import *


# copy auto pilot code
# add options to make the environment empty
parser = argparse.ArgumentParser()

parser.add_argument('--version', type=int, choices=[10,11], default=11)
parser.add_argument('--num_steps', type=int, default=)
parser.add_argument('--scenarios', action='store_true')
parser.add_argument('--empty', action='store_true')
parser.add_argument('--split', type=str, default='training', 
        choices=['debug','devtest','testing','training'])
parser.add_argument('--routenum', type=int)

routes = f'routes_{args.split}'
if args.routenum:
    routes = f'{routes}/route_{args.routenum:02d}'
routes = f'{routes}.xml'
scenarios = 'all_towns_traffic_scenarios_public.json' if args.scenarios \
        else 'no_traffic_scenarios.json'

date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
suffix = f'debug/{date_str}' if args.debug else f'{date_str}' 
save_root = f'/data/leaderboard/data/rl/semantic_enc/{suffix}'

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
        }

agent_config = {

        }
