import os, sys, time, signal
import yaml, json
import argparse
import traceback
import numpy as np

from tqdm import tqdm
from datetime import datetime
from env import CarlaEnv
from carla import Client
from agent import WaypointAgent

from team_code.common.utils import dict_to_sns
from stable_baselines.common.schedules import get_schedule_fn
from stable_baselines.common import tf_util, SetVerbosity, TensorboardWriter


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
    sac_log = episode_log['sac']
    rewards = []

    model = agent.model

    # LOGGING STUFF
    new_tb_log = model._init_num_timesteps(False) #reset_num_timesteps=False
    callback = model._init_callback(None) # callback=None

    with SetVerbosity(1), TensorboardWriter(model.graph, model.tensorboard_log, 'SAC', new_tb_log) as writer:

        model._setup_learn()
        lr = get_schedule_fn(model.learning_rate)
        current_lr = lr(1)

        episode_rewards = [0.0]
        episode_successes = []
        obs = env.reset(log=episode_log)

        n_updates = 0
        infos_values=[]

        print(f'begin training, logs at: {agent.tensorboard_root}')
        callback.on_training_start(locals(), globals())
        callback.on_rollout_start()

        for step in tqdm(range(begin_step, config.agent.total_timesteps)):

            # get SAC prediction, step the env
            burn_in = (step - begin_step) < config.agent.burn_timesteps # exploration
            agent.set_burn_in(burn_in)
            agent.set_deterministic(False)

            new_obs, reward, done, info = env.step(None)
            sac_log['total_steps'] += 1

            model.num_timesteps += 1
                   
            callback.update_locals(locals())
            if callback.on_step() is False:
                break

            # store in replay buffer
            #if sac_log['total_steps'] > 60: # 3 seconds of warmup time @ 20Hz
            #model.replay_buffer_add(obs, action, reward, new_obs, float(done), {})
            _obs, _action, _reward, _new_obs, _done, _info = env.exp
            model.replay_buffer_add(_obs, _action, _reward, _new_obs, _done, _info)
            obs = new_obs

            if writer is not None:
                ep_reward = np.array([reward]).reshape((1,-1))
                ep_done = np.array([done]).reshape((1,-1))
                tf_util.total_episode_reward_logger(model.episode_reward, 
                        ep_reward, ep_done, writer, model.num_timesteps)

            # train at this timestep if applicable
            if step % config.agent.train_frequency == 0:
                callback.on_rollout_end()
                mb_info_vals = []

                
                for grad_step in range(config.agent.gradient_steps):

                    if burn_in or not model.replay_buffer.can_sample(model.batch_size):
                        break
                    n_updates += 1

                    # policy and value network update
                    frac = 1.0 - step/config.agent.total_timesteps
                    current_lr = lr(frac)
                    train_vals = model._train_step(step, writer, current_lr)
                    mb_info_vals.append(train_vals)
                    policy_loss, _, _, value_loss, entropy, _, _ = train_vals

                    sac_log['mean_policy_loss'] += policy_loss
                    sac_log['mean_value_loss'] += value_loss
                    sac_log['mean_entropy'] += entropy

                    # target network update
                    if step % config.agent.target_update_interval == 0:
                        model.sess.run(model.target_update_op)

                    if config.agent.verbose and step % config.agent.log_frequency == 0:
                        write_str = f'\nstep {step}\npolicy_loss = {policy_loss:.3f}\nvalue_loss = {value_loss:.3f}\nentropy = {entropy:.3f}'
                        tqdm.write(write_str)

                if len(mb_info_vals) > 0:
                    infos_values = np.mean(mb_info_vals, axis=0)

                callback.on_rollout_start()

            # save model if applicable
            if step % config.agent.save_frequency == 0 and not burn_in:
                weights_path = f'{config.save_root}/weights/{step:07d}'
                model.save(weights_path)

            episode_rewards[-1] += reward
            if done:
                # record then reset metrics
                episode_steps = sac_log['total_steps']
                sac_log['mean_policy_loss'] /= episode_steps
                sac_log['mean_value_loss'] /= episode_steps
                sac_log['mean_entropy'] /= episode_steps
                sac_log['total_reward'] = episode_rewards[-1]
                log['checkpoints'].append(episode_log)

                #with open(f'{config.save_root}/logs/log.json', 'w') as f:
                #    json.dump(log, f, indent=4, sort_keys=False)
                #with open(f'{config.save_root}/logs/rewards/{episode_idx:06d}.npy', 'wb') as f:
                #    np.save(f, rewards)

                with open(f'{config.save_root}/logs/log.json', 'w') as f:
                    json.dump(log, f, indent=4, sort_keys=False)


                episode_rewards.append(0.0)
                maybe_is_success = info.get('is_success')
                if maybe_is_success is not None:
                    episode_successes.append(float(maybe_is_success))

                if len(episode_rewards[-101:-1]) == 0:
                    mean_reward = -np.inf
                else:
                    nans = np.isnan(episode_rewards[-101:1])
                    if np.any(nans):
                        print(f'found {np.sum(nans)} NaNs')
                    mean_reward = round(float(np.mean(episode_rewards[-101:-1])), 1)

                num_episodes = len(episode_rewards) - 1

                # cleanup and reset
                env.cleanup()
                episode_idx += 1
                episode_log = setup_episode_log(episode_idx)
                obs = env.reset(log=episode_log)
                sac_log = episode_log['sac']
                rewards = []

        #print('done training')
        callback.on_training_end()
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
