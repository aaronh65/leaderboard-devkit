import os
import cv2
import argparse

def main(args):
    
    image_dir = os.path.join(args.target_dir, 'images')
    assert os.path.exists(image_dir), 'no images found - exiting'
    video_dir = os.path.join(args.target_dir, 'videos')
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)

    width, height = None, None
    episodes = sorted(os.listdir(image_dir))
    for ep in episodes:

        input_dir = os.path.join(image_dir, ep)
        if len(os.listdir(input_dir)) <= 1:
            # avoids crashes when you open a video with one frame in it
            continue

        save_path = os.path.join(video_dir, f'{ep}.mp4')
        if os.path.exists(save_path):
            continue

        if width is None or height is None:
            fname = os.listdir(input_dir)[0]
            impath = os.path.join(input_dir, fname)
            im = cv2.imread(impath)
            height, width, channels = im.shape

        if width % 2 == 1 or height % 2 == 1:
            cmd = f'ffmpeg -y -r {args.fps} -s {width}x{height} -f image2 -i {input_dir}/%06d.png -pix_fmt yuv420p -vf \"pad=ceil(iw/2)*2:ceil(ih/2)*2\" {save_path}'
        else:
            cmd = f'ffmpeg -y -r {args.fps} -s {width}x{height} -f image2 -i {input_dir}/%06d.png -pix_fmt yuv420p {save_path}'
        os.system(cmd)

def parse_args():
    parser = argparse.ArgumentParser()
    # /home/aaron/workspace/carla/2020_CARLA_challenge/leaderboard/results/rl/waypoint_agent/video_test
    parser.add_argument('--target_dir', type=str, required=True)
    parser.add_argument('--fps', type=int, default=10)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)
