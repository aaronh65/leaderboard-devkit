import argparse, traceback
import yaml, tqdm

from carla import Client
from data_agent import AutoPilot
from data_env import CarlaEnv
from team_code.common.utils import dict_to_sns
from team_code.rl.common.env_utils import *

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

def perturb_transform(actor):
    dyaw = np.random.randint(90) - 45
    dx, dy = np.random.randint(3, size=2) - 1.5
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
        env = CarlaEnv(config, client, agent)
        env.reset()
        for step in tqdm.tqdm(range(config.env.num_steps)):
            _, _, done, _ = env.step(None)

            if done:
                env.cleanup()
                env.reset()

            if step % 100 == 0:
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
