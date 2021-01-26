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
    plt.savefig(f'{args.target_dir}/total_rewards.png')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_dir', type=str, required=True)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)
