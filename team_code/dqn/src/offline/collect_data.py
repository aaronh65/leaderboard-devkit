import os, time, yaml, json
import argparse
import traceback
from datetime import datetime
from tqdm import tqdm

from carla import Client
from pathlib import Path
from dqn.src.env.env import CarlaEnv
from dqn.src.agents.privileged_agent import PrivilegedAgent

from misc.utils import dict_to_sns, port_in_use

#def collect(config, agent, env):
#
#    # start environment and run
#    status = env.reset()
#    while status != 'done':
#        reward, done, info = env.step()
#        if done:
#            env.cleanup()
#            status = env.reset()

def main(args, config):

    agent = PrivilegedAgent(f'{config.save_root}/config.yml')

    client = Client('localhost', args.world_port)
    client.set_timeout(600)
    try:
        env = CarlaEnv(config, client, agent)
        env.reset()
        env.rollout(complete=True)
        #collect(config, agent, env)
    except KeyboardInterrupt:
        env.cleanup()
        print('caught KeyboardInterrupt')
    except Exception as e:
        env.cleanup()
        traceback.print_exception(type(e), e, e.__traceback__)
    finally:
        del env

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-D', '--debug', action='store_true')
    parser.add_argument('--data_root', type=str, default='/data/aaronhua')
    parser.add_argument('--id', type=str, default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument('--save_debug', action='store_true')
    parser.add_argument('--save_data', action='store_true', default=True)
    parser.add_argument('--short_stop', action='store_true')
    parser.add_argument('--dagger_expert', action='store_true')
    parser.add_argument('--data_hack', action='store_true', default=True)


    # server/client stuff
    parser.add_argument('-WP', '--world_port', type=int, default=2000)
    parser.add_argument('-TP', '--traffic_port', type=int, default=8000)
    parser.add_argument('-TS', '--traffic_seed', type=int, default=0)

    # setup
    parser.add_argument('--split', type=str, default='devtest', 
            choices=['debug', 'devtest', 'testing', 'training'])
    parser.add_argument('--routenum', type=int) # optional
    parser.add_argument('--scenarios', action='store_true', default=True) # leaderboard scenarios
    parser.add_argument('--repetitions', type=int, default=1)
    parser.add_argument('--empty', action='store_true') # other agents present?
    parser.add_argument('--random', action='store_true')
    parser.add_argument('--forward', action='store_true')

    args = parser.parse_args()

    assert not port_in_use(args.traffic_port), \
        f'traffic manager port {args.traffic_port} already in use!!'

    # basic setup
    project_root = os.environ['PROJECT_ROOT']
    suffix = f'debug/{args.id}' if args.debug else args.id
    save_root = Path(f'{args.data_root}/leaderboard/data/dqn/{suffix}')
    save_root.mkdir(parents=True, exist_ok=True)
    
    # environment setup
    routes = f'routes_{args.split}'
    if args.routenum:
        routes = f'{routes}/route_{args.routenum:02d}'
    routes = f'{routes}.xml'
    scenarios = 'all_towns_traffic_scenarios_public.json' if args.scenarios \
            else 'no_traffic_scenarios.json'

    # get template config
    config_path = f'{project_root}/team_code/dqn/config/privileged_agent.yml'
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)

    os.environ["DQN_COLLECT"] = "1"
    
    # setup config
    config['project_root'] = project_root
    config['save_root'] = str(save_root)
    config['save_data'] = args.save_data
    config['save_debug'] = args.save_debug

    # setup env config
    econf = config['env']
    econf['world_port'] = args.world_port
    econf['trafficmanager_port'] = args.traffic_port
    econf['trafficmanager_seed'] = args.traffic_seed
    econf['routes'] = routes
    econf['scenarios'] = scenarios
    econf['repetitions'] = args.repetitions
    econf['empty'] = args.empty
    econf['random'] = args.random
    econf['short_stop'] = args.short_stop

    aconf = config['agent']
    aconf['mode'] = 'data'
    aconf['dagger_expert'] = args.dagger_expert
    aconf['data_hack'] = args.data_hack
    aconf['forward'] = args.forward
 
    # save new config
    with open(f'{save_root}/config.yml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    #config['save_root'] = save_root
    config = dict_to_sns(config)
    config.save_root = save_root
    config.env = dict_to_sns(config.env)
    config.agent = dict_to_sns(config.agent)

    return args, config

if __name__ == '__main__':
    args, config = parse_args()
    main(args, config)
