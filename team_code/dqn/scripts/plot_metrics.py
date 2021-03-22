import os
import argparse
import json
import matplotlib.pyplot as plt
import numpy as np

def main(args):

    with open(f'{args.target_dir}/logs/log.json', 'r') as f:
        log = json.load(f)['checkpoints']
    rewards = []
    for episode_ckpt in log:
        rewards.append(episode_ckpt['sac']['total_reward'])

    episodes = np.arange(len(rewards))
    plt.plot(episodes, rewards)
    plt.xlabel('episode #')
    plt.ylabel('episode total reward')
    plt.title ('training episode rewards')
    #plt.show()

    reward_base = f'{args.target_dir}/logs/rewards'
    if not os.path.exists(f'{reward_base}/plots'):
        os.makedirs(f'{reward_base}/plots')
    plt.savefig(f'{reward_base}/plots/total_rewards.png')
    plt.clf()

    for name in os.listdir(f'{reward_base}/data'):
        rewards = np.load(f'{reward_base}/data/{name}')
        time = np.arange(rewards.shape[0])
        plt.plot(time, rewards)
        plt.xlabel('frame')
        plt.ylabel('reward')
        plt.title('reward per frame')
        plt.savefig(f'{reward_base}/plots/{name[:-4]}.png')
        plt.clf()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_dir', type=str, required=True)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)
