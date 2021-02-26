import sys, time
import yaml, json
import argparse
import traceback

from tqdm import tqdm
from env import CarlaEnv
from carla import Client
from agent import DSPredAgent

from common.utils import dict_to_sns

import cv2

def setup_episode_log(episode_idx):

    episode_log = {}
    episode_log['index'] = episode_idx
    agent_log = {}
    agent_log['total_steps'] = 0
    agent_log['total_reward'] = 0
    episode_log['agent'] = agent_log
    return episode_log

def train(config, agent, env):

    begin_step = 0
    episode_idx = 0
    episode_rewards = [0.0]

    log = {'checkpoints': []}
    episode_log = setup_episode_log(episode_idx)
    agent_log = episode_log['agent']

    # start environment and run
    
    env.reset(log=episode_log)
    for step in tqdm(range(begin_step, config.agent.total_timesteps)):

        # burn in?
        reward, done = env.step() # epsilon greedy? deterministic? 
        agent_log['total_steps'] += 1

        if step > 60 + config.agent.batch_size:
            batch = env.buf.sample()
            #cv2.imshow('topdown', exp[0][0][0])
            #cv2.waitKey(1)
            pass
               
        # save model if applicable
        episode_rewards[-1] += reward
        if done:
            # record then reset metrics
            agent_log['total_reward'] = episode_rewards[-1]
            log['checkpoints'].append(episode_log)

            with open(f'{config.save_root}/logs/log.json', 'w') as f:
                json.dump(log, f, indent=4, sort_keys=False)

            episode_rewards.append(0.0)

            # cleanup and reset
            env.cleanup()
            episode_idx += 1
            episode_log = setup_episode_log(episode_idx)
            env.reset(log=episode_log)
            agent_log = episode_log['agent']

        # train model
        if True: # can update to wait some number of steps
            # one gradient step for now
            pass

def main(args):
    client = Client('localhost', 2000)
    client.set_timeout(600)

    # get configs and spin up agent
    with open(args.config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config = dict_to_sns(config)
    config.env = dict_to_sns(config.env)
    config.agent = dict_to_sns(config.agent)

    try:
        agent = DSPredAgent(config)
        env = CarlaEnv(config, client, agent)
        train(config, agent, env)
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
