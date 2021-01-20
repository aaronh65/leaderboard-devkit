import os
import sys
sys.path.append('../common')
import argparse
from utils import mkdir_if_not_exists

def main(args):
    image_dir = os.path.join(args.target_dir, 'images')
    assert os.path.exists(image_dir), 'no images found - exiting'
    video_dir = os.path.join(args.target_dir, 'videos')
    mkdir_if_not_exists(video_dir)

    episodes = sorted(os.listdir(image_dir))
    for ep in episodes:
        input_dir = os.path.join(image_dir, ep)
        impaths = [os.path.join(input_dir, fname) for fname in sorted(os.listdir(input_dir))]
        if len(os.listdir(input_dir)) <= 1:
            # avoids crashes when you open a video with one frame in it
            continue

        save_path = os.path.join(video_dir, f'{ep}.mp4')
        if os.path.exists(save_path):
            continue
        cmd = f'ffmpeg -y -r 2 -s 256x256 -f image2 -i {input_dir}/%06d.png -pix_fmt yuv420p {save_path}'
        os.system(cmd)

def parse_args():
    parser = argparse.ArgumentParser()
    # /home/aaron/workspace/carla/2020_CARLA_challenge/leaderboard/results/rl/waypoint_agent/video_test
    parser.add_argument('--target_dir', type=str, required=True)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)
