import argparse, traceback
import yaml

from carla import Client
from data_agent import AutoPilot
from team_code.rl.common.base_env import BaseEnv
from team_code.common.utils import dict_to_sns
from tqdm import tqdm
from team_code.rl.common.env_utils import *

def perturb_transform(actor):
    dyaw = np.random.randint(30) - 15
    dx, dy = np.random.randint(2, size=2) - 1
    transform = actor.get_transform()
    perturbed = add_transform(transform, dx=dx, dy=dy, dyaw=dyaw)
    actor.set_transform(perturbed)

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
        for step in tqdm(range(config.env.num_steps)):
            env.step(None)

            if step % 10 == 0:
                perturb_transform(env.hero_actor)

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
