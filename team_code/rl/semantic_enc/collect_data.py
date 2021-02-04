import argparse

from carla import Client
from data_agent import AutoPilot
from team_code.common.base_env import BaseEnv

def main(args):
    with open(args.config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config = dict_to_sns(config)
    config.env = dict_to_sns(config.env)
    config.sac = dict_to_sns(config.sac)

    client = Client('localhost', 2000)
    agent = AutoPilot(config)
    env = BaseEnv(config, client, agent)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)
