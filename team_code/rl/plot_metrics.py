import argparse
import matplotlib.pyplot as plt
import numpy as np

def main(args):

    metrics = []
    metric_names = ['rewards', 'policy_losses', 'value_losses', 'entropies']
    for name in metric_names:
        with open(f'{args.target_dir}/{name}.npy', 'rb') as f:
            metrics.append(np.load(f))

    rewards, _, _, _ = metrics
    episodes = np.arange(len(rewards))
    plt.plot(episodes, rewards)
    plt.xlabel('episode #')
    plt.ylabel('episode reward')
    plt.title ('training episode rewards')
    #plt.show()
    plt.savefig(f'{args.target_dir}/result.png')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_dir', type=str, required=True)
    parser.add_argument('--load_from', type=str)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)
