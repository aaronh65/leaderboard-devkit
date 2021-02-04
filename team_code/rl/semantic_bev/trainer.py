import os, sys, time, signal
import yaml, json, pickle
import argparse
import traceback
import numpy as np
from datetime import datetime
from tqdm import tqdm

from env import CarlaEnv
from carla import Client
from agent import WaypointAgent

from team_code.common.utils import dict_to_sns

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

def setup_episode_log(episode_idx):

    episode_log = {}
    episode_log['index'] = episode_idx
    sac_log = {}
    sac_log['total_steps'] = 0
    sac_log['total_reward'] = 0
    sac_log['mean_policy_loss'] = 0
    sac_log['mean_value_loss'] = 0
    sac_log['mean_entropy'] = 0
    episode_log['sac'] = sac_log
    return episode_log

def restore(config):
    with open(f'{config.save_root}/logs/log.json', 'r') as f:
        log = json.load(f)
    weight_names = sorted(os.listdir(f'{config.save_root}/weights'))
    weight_last = weight_names[-1].split('.')[0]
    begin_step = int(weight_last)
    return log, begin_step

def train(config, agent, env):

    if RESTORE:
        log, begin_step = restore(config)
        episode_idx = log['checkpoints'][-1]['index'] + 1
    else:
        log = {'checkpoints': []}
        begin_step = 0
        episode_idx = 0

    # start environment and run
    episode_log = setup_episode_log(episode_idx)
    obs = env.reset(log=episode_log)
    sac_log = episode_log['sac']
    rewards = []

    for step in tqdm(range(begin_step, config.sac.total_timesteps)):

        # get SAC prediction, step the env
        burn_in = (step - begin_step) < config.sac.burn_timesteps # exploration
        agent.set_burn_in(burn_in)
        #action = agent.predict(obs, burn_in=burn_in)
        new_obs, reward, done, info = env.step(None)
        action = agent.cached_action
        sac_log['total_steps'] += 1

        # store in replay buffer
        rewards.append(reward)
        sac_log['total_reward'] += reward
        if sac_log['total_steps'] > 60: # 3 seconds of warmup time @ 20Hz
            _obs, _action, _reward, _new_obs, _done, _ = env.exp
            agent.model.replay_buffer_add(_obs, _action, _reward, _new_obs, _done, _)
                
        # train at this timestep if applicable
        if step % config.sac.train_frequency == 0 and not burn_in:
            mb_info_vals = []
            for grad_step in range(config.sac.gradient_steps):

                # policy and value network update
                frac = 1.0 - step/config.sac.total_timesteps
                lr = agent.model.learning_rate*frac
                train_vals = agent.model._train_step(step, None, lr)
                policy_loss, _, _, value_loss, entropy, _, _ = train_vals

                sac_log['mean_policy_loss'] += policy_loss
                sac_log['mean_value_loss'] += value_loss
                sac_log['mean_entropy'] += entropy

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

        if done:
            # record then reset metrics
            episode_steps = sac_log['total_steps']
            sac_log['mean_policy_loss'] /= episode_steps
            sac_log['mean_value_loss'] /= episode_steps
            sac_log['mean_entropy'] /= episode_steps

            log['checkpoints'].append(episode_log)
            with open(f'{config.save_root}/logs/log.json', 'w') as f:
                json.dump(log, f, indent=4, sort_keys=False)
            with open(f'{config.save_root}/logs/rewards/{episode_idx:06d}.npy', 'wb') as f:
                np.save(f, rewards)
            with open(f'{config.save_root}/logs/replay_buffer.pkl', 'wb') as f:
                pickle.dump(agent.model.replay_buffer, f)

            # cleanup and reset
            env.cleanup()
            episode_idx += 1
            episode_log = setup_episode_log(episode_idx)
            obs = env.reset(log=episode_log)
            sac_log = episode_log['sac']
            rewards = []

        else:
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
