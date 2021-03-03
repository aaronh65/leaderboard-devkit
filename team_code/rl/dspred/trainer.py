import os, sys, time
import yaml, json
import argparse
import traceback
from datetime import datetime

from carla import Client 
from env import CarlaEnv
from agent import DSPredAgent
from online_map_model import MapModel

from common.utils import dict_to_sns
import pytorch_lightning as pl


def train(config, agent, env):
   
    env.reset()
    for step in tqdm(range(begin_step, config.agent.total_timesteps)):

        # burn in?
        reward, done = env.step() # epsilon greedy? deterministic? 

        if step > 60 + config.agent.batch_size:
            batch = env.buf.sample()
               
        # save model if applicable
        episode_rewards[-1] += reward
        if done:
            # cleanup and reset
            env.cleanup()
            env.reset()

        # train model
        if True: # can update to wait some number of steps
            # one gradient step for now
            pass

def main(args, config):

    try:
        agent = DSPredAgent(config)
        client = Client('localhost', 2000)
        client.set_timeout(600)
        env = CarlaEnv(config, client, agent)
        model = agent.net
        model.env = env
        status = model.env.reset()
        for i in range(10):
            print(i)
            model.env.step()
        #train(config, agent, env)
    except KeyboardInterrupt:
        print('caught KeyboardInterrupt')
    except Exception as e:
        traceback.print_exception(type(e), e, e.__traceback__)
    finally:
        env.cleanup()
        del env

def parse_args():

    # assert to make sure setup.bash sourced?
    #project_root = os.environ['PROJECT_ROOT']
    project_root = '/home/aaron/workspace/carla/leaderboard-devkit'

    # retrieve template config
    config_path = f'{project_root}/team_code/rl/config/dspred.yml'
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config['project_root'] = project_root
    #config['carla_root'] = os.environ['CARLA_ROOT']

    # query new arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-D', '--debug', action='store_true')
    parser.add_argument('-G', '--gpu', type=int, default=0)
    parser.add_argument('--data_root', type=str, default='/data')
    parser.add_argument('--split', type=str, default='training', choices=['debug', 'devtest', 'testing', 'training'])
    #parser.add_argument('--routenum', type=int) # optional
    parser.add_argument('--no_scenarios', action='store_true') # leaderboard-triggered scnearios
    #parser.add_argument('--repetitions', type=int, default=1) # should we directly default to this in indexer?
    parser.add_argument('--empty', action='store_true')
    parser.add_argument('--buffer_size', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()

    # modify template config
    econf = config['env']
    scenarios = 'no_traffic_scenarios.json' if args.no_scenarios \
            else 'all_towns_traffic_scenarios_public.json'
    econf['scenarios'] = scenarios
    econf['empty'] = args.empty

    aconf = config['agent']
    total_timesteps = 2000 if args.debug else aconf['total_timesteps']
    burn_timesteps = 250 if args.debug else 2000
    save_frequency = 500 if args.debug else 5000

    aconf['mode'] = 'train'
    aconf['total_timesteps'] = total_timesteps
    aconf['burn_timesteps'] = burn_timesteps
    aconf['save_frequency'] = save_frequency

    # save new config path
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f'debug/{date_str}' if args.debug else date_str
    save_root = f'{args.data_root}/leaderboard/results/rl/dspred/{suffix}'
    os.makedirs(save_root)
    with open(f'{save_root}/config.yml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    config = dict_to_sns(config)
    config.env = dict_to_sns(config.env)
    config.agent = dict_to_sns(config.agent)
    return args, config

if __name__ == '__main__':
    args, config = parse_args()
    main(args, config)
