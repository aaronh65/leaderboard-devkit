import signal
import os, sys, time, yaml
import argparse
import traceback
import numpy as np
from tqdm import tqdm
from datetime import datetime

from team_code.common.utils import dict_to_sns
from waypoint_agent import WaypointAgent
from env import CarlaEnv
from carla import Client

RESTORE = int(os.environ.get("RESTORE", 0))

# setup metrics
def setup(config):

    metric_names = ['rewards', 'policy_losses', 'value_losses', 'entropies']
    if RESTORE:
        weight_names = sorted(os.listdir(f'{config.save_root}/weights'))
        weight_last = weight_names[-1].split('.')[0]
        begin_step = int(weight_last)

        metric_arrs = []
        for name in metric_names:
            with open(f'{config.save_root}/{name}.npy', 'rb') as f:
                metric_arrs.append(np.load(f).tolist())
        metrics = {name: arr for name, arr in zip(metric_names, metric_arrs)}
    else:
        begin_step = 0
        metrics = {name: [] for name in metric_names}

    return begin_step, metrics

def train(config, agent, env):

    begin_step, metrics = setup(config)
    print(begin_step)
    print(len(metrics['rewards']))

    # per episode counts
    total_reward = 0
    total_policy_loss = 0
    total_value_loss = 0
    total_entropy = 0
    episode_steps = 0

    # start environment and run
    obs = env.reset()
    for step in tqdm(range(begin_step, config.sac.total_timesteps)):
        
        # random exploration at the beginning
        burn_in = (step - begin_step) < config.sac.burn_timesteps

        # get SAC prediction, step the env
        action = agent.predict(obs, burn_in=burn_in)
        new_obs, reward, done, info = env.step(action)

        # store in replay buffer
        if env.frame > 60: # 3 seconds of warmup time @ 20Hz
            agent.model.replay_buffer.add(obs, action, reward, new_obs, float(done))
        total_reward += reward
        episode_steps += 1

        if done:

            # record then reset metrics
            metrics['rewards'].append(total_reward)
            metrics['policy_losses'].append(total_policy_loss/episode_steps)
            metrics['value_losses'].append(total_value_loss/episode_steps)
            metrics['entropies'].append(total_entropy/episode_steps)

            for name, arr in metrics.items():
                save_path = f'{config.save_root}/{name}.npy'
                with open(save_path, 'wb') as f:
                    np.save(f, arr)

            total_reward = 0
            total_policy_loss = 0
            total_value_loss = 0
            total_entropy = 0
            episode_steps = 0

            # cleanup and reset
            env.cleanup()
            obs = env.reset()
        
        # train at this timestep if applicable
        if step % config.sac.train_frequency == 0 and not burn_in:
            mb_info_vals = []
            for grad_step in range(config.sac.gradient_steps):

                # policy and value network update
                frac = 1.0 - step/config.sac.total_timesteps
                lr = agent.model.learning_rate*frac
                train_vals = agent.model._train_step(step, None, lr)
                policy_loss, _, _, value_loss, entropy, _, _ = train_vals

                total_policy_loss += policy_loss
                total_value_loss += value_loss
                total_entropy += entropy

                # target network update
                if step % config.sac.target_update_interval == 0:
                    agent.model.sess.run(agent.model.target_update_op)

                if config.sac.verbose and step % config.sac.log_frequency == 0:
                    write_str = f'\nstep {step}\npolicy_loss = {policy_loss:.3f}\nvalue_loss = {value_loss:.3f}\nentropy = {entropy:.3f}'
                    tqdm.write(write_str)

        # save model if applicable
        if step % config.sac.save_frequency == 0 and not burn_in:
            weights_path = f'{config.save_root}/weights/{step:07d}'
            agent.model.save(weights_path)

        obs = new_obs

    #print('done training')

def main(args):
    client = Client('localhost', 2000)

    # get configs and spin up agent
    with open(args.config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config = dict_to_sns(config)
    config.env = dict_to_sns(config.env)
    config.sac = dict_to_sns(config.sac)

    try:
        agent = WaypointAgent(config)
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
