import os, time
import yaml, json
import argparse
import traceback
from datetime import datetime

from pathlib import Path
from carla import Client 
from env import CarlaEnv
from agent import DSPredAgent
from map_model import MapModel

from common.utils import dict_to_sns, port_in_use
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from rl.dspred.global_buffer import ReplayBuffer

def main(args, config):
    try:

        client = Client('localhost', config.env.world_port)
        client.set_timeout(600)
        agent = DSPredAgent(config)
        env = CarlaEnv(config, client, agent)

        ReplayBuffer.setup(
                config.agent.buffer_size, 
                config.agent.batch_size,
                config.agent.n)

        model = agent.net
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
    parser.add_argument('--split', type=str, default='training', choices=['debug', 'devtest', 'testing', 'training'])
    parser.add_argument('--save_debug', action='store_true')
    parser.add_argument('--save_data', action='store_true')

    # server/client stuff
    parser.add_argument('-WP', '--world_port', type=int, default=2000)
    parser.add_argument('-TP', '--traffic_port', type=int, default=8000)
    parser.add_argument('-TS', '--traffic_seed', type=int, default=0)

    #parser.add_argument('--routenum', type=int) # optional
    parser.add_argument('--no_scenarios', action='store_true') # leaderboard-triggered scnearios
    #parser.add_argument('--repetitions', type=int, default=1) # should we directly default to this in indexer?
    parser.add_argument('--empty', action='store_true')
    parser.add_argument('--buffer_size', type=int, default=50000)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--log', action='store_true')
    parser.add_argument('--save_dir', type=Path, default='checkpoints')
    parser.add_argument('--id', type=str, default=datetime.now().strftime("%Y%m%d_%H%M%S"))

    parser.add_argument('--n', type=int, default=20)
    parser.add_argument('--epsilon', type=float, default=0.1)
    parser.add_argument('--rollout_steps', type=int, default=20)
    parser.add_argument('--forward', action='store_true')
    args = parser.parse_args()

    assert not port_in_use(args.traffic_port), \
            f'traffic manager port {args.traffic_port} already in use!!'

    # assert to make sure setup.bash sourced?
    project_root = os.environ['PROJECT_ROOT']
    suffix = f'debug/{args.id}' if args.debug else args.id
    save_root = args.data_root / f'leaderboard/results/rl/dspred/{suffix}'
    args.save_dir = save_root / args.save_dir
    args.save_dir.mkdir(parents=True, exist_ok=True)

    # retrieve template config
    config_path = f'{project_root}/team_code/rl/config/dspred.yml'
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)

    config['project_root'] = project_root
    config['save_root'] = save_root
    config['save_data'] = args.save_data
    config['save_debug'] = args.save_debug

    # modify template config
    econf = config['env']
    econf['world_port'] = args.world_port
    econf['trafficmanager_port'] = args.traffic_port
    econf['trafficmanager_seed'] = args.traffic_seed
    scenarios = 'no_traffic_scenarios.json' if args.no_scenarios \
            else 'all_towns_traffic_scenarios_public.json'
    econf['scenarios'] = scenarios
    econf['empty'] = args.empty
    econf['random'] = True

    aconf = config['agent']
    total_timesteps = 200 if args.debug else aconf['total_timesteps']
    #burn_timesteps= 1000
    burn_timesteps = 50
    save_frequency = 500 if args.debug else 5000
    batch_size = 4 if args.debug else args.batch_size
    buffer_size = 1000 if args.debug else args.buffer_size

    aconf['mode'] = 'train'
    aconf['total_timesteps'] = total_timesteps
    aconf['burn_timesteps'] = burn_timesteps
    aconf['save_frequency'] = save_frequency
    aconf['batch_size' ] = batch_size
    aconf['buffer_size'] = buffer_size
    aconf['epsilon'] = args.epsilon
    aconf['n'] = args.n
    aconf['rollout_steps'] = args.rollout_steps
    aconf['forward'] = args.forward

    # save new config
    with open(f'{save_root}/config.yml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    config = dict_to_sns(config)
    config.env = dict_to_sns(config.env)
    config.agent = dict_to_sns(config.agent)

    return args, config

if __name__ == '__main__':
    args, config = parse_args()
    main(args, config)
