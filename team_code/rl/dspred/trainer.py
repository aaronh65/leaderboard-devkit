import os, sys, time
import yaml, json
import argparse
import traceback
from datetime import datetime
import pathlib

from carla import Client 
from env import CarlaEnv
from agent import DSPredAgent
from online_map_model import MapModel

from common.utils import dict_to_sns
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

def main(args, config):
    try:
        agent = DSPredAgent(config)
        client = Client('localhost', 2000)
        client.set_timeout(600)
        env = CarlaEnv(config, client, agent)
        model = agent.net
        model.env = env
        model.setup_train(env, config)

        logger = False 
        if args.log:
            logger = WandbLogger(id=args.id, save_dir=str(args.save_dir), project='dqn')

        checkpoint_callback = ModelCheckpoint(config.save_root, save_top_k=1)
        trainer = pl.Trainer(
            gpus=[args.gpu], 
            max_steps=config.agent.total_timesteps,
            val_check_interval=10,
            logger=logger, 
            checkpoint_callback=checkpoint_callback)
        trainer.fit(model)

    except KeyboardInterrupt:
        print('caught KeyboardInterrupt')
    except Exception as e:
        traceback.print_exception(type(e), e, e.__traceback__)
    finally:
        env.cleanup()
        del env

def parse_args():

    # query new arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-D', '--debug', action='store_true')
    parser.add_argument('-G', '--gpu', type=int, default=0)
    parser.add_argument('--data_root', type=str, default='/data')
    parser.add_argument('--remote', action='store_true')
    parser.add_argument('--split', type=str, default='training', choices=['debug', 'devtest', 'testing', 'training'])
    #parser.add_argument('--routenum', type=int) # optional
    parser.add_argument('--no_scenarios', action='store_true') # leaderboard-triggered scnearios
    #parser.add_argument('--repetitions', type=int, default=1) # should we directly default to this in indexer?
    parser.add_argument('--empty', action='store_true')
    parser.add_argument('--buffer_size', type=int, default=50000)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--log', action='store_true')
    parser.add_argument('--save_dir', type=pathlib.Path, default='checkpoints')
    parser.add_argument('--id', type=str, default=datetime.now().strftime("%Y%m%d_%H%M%S")) # replace with datetime
    args = parser.parse_args()

    # assert to make sure setup.bash sourced?
    project_root = os.environ['PROJECT_ROOT']
    #prefix = '/home/aaron/workspace/carla' if not args.remote else '/home/aaronhua'
    #project_root = f'{prefix}/leaderboard-devkit'
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f'debug/{date_str}' if args.debug else date_str
    save_root = f'{args.data_root}/leaderboard/results/rl/dspred/{suffix}'
    os.makedirs(save_root)

    args.save_dir = save_root / args.save_dir
    args.save_dir.mkdir(parents=True, exist_ok=True)

    # retrieve template config
    config_path = f'{project_root}/team_code/rl/config/dspred.yml'
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config['project_root'] = project_root
    config['save_root'] = save_root
    #config['carla_root'] = os.environ['CARLA_ROOT']

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
    batch_size = 4 if args.debug else args.batch_size
    buffer_size = 200 if args.debug else args.buffer_size

    aconf['mode'] = 'train'
    aconf['total_timesteps'] = total_timesteps
    aconf['burn_timesteps'] = burn_timesteps
    aconf['save_frequency'] = save_frequency
    aconf['batch_size' ] = batch_size
    aconf['buffer_size'] = buffer_size

    # save new config path
    with open(f'{save_root}/config.yml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    config = dict_to_sns(config)
    config.env = dict_to_sns(config.env)
    config.agent = dict_to_sns(config.agent)
    return args, config

if __name__ == '__main__':
    args, config = parse_args()
    main(args, config)
