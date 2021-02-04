import argparse
import yaml

from carla import Client
from data_agent import AutoPilot
from team_code.rl.common.base_env import BaseEnv
from team_code.common.utils import dict_to_sns

def main(args):
    client = Client('localhost', 2000)

    with open(args.config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config = dict_to_sns(config)
    config.env = dict_to_sns(config.env)
    config.agent = dict_to_sns(config.agent)

    try:
        agent = AutoPilot(config)
        env = BaseEnv(config, client, agent)
        env.reset()
        for step in range(config.env.num_steps):
            pass
    except KeyboardInterrupt:
        print('caught KeyboardInterrupt')
    except Exception as e:
        traceback.print_exception(type(e), e, e.__traceback__)

    finally:
        env.cleanup()
        del env

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)
