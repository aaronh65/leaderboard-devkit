import os
import argparse

def main(args):
    splits = os.listdir(args.target_dir)
    for split in splits:
        video_dir = os.path.join(args.target_dir, split, 'videos')
        image_dir = os.path.join(args.target_dir, split, 'images')
        if not os.path.exists(image_dir):
            continue
        routes = sorted(os.listdir(image_dir))
        for route in routes:
            route_image_dir = os.path.join(image_dir, route)
            input_dirs = [os.path.join(route_image_dir, dir_name) for dir_name in sorted(os.listdir(route_image_dir))]
            input_dirs = [dir_name for dir_name in input_dirs if 'bkup' not in dir_name and 'mp4' not in dir_name]
            for input_dir in input_dirs:
                
                # get save_path
                tokens = input_dir.split('/')
                route, repetition = tokens[-2:]
                route_video_dir = os.path.join(video_dir, route)
                if not os.path.exists(route_video_dir):
                    os.makedirs(route_video_dir)
                save_path = os.path.join(route_video_dir, f'{repetition}.mp4')

                # make video
                cmd = f'ffmpeg -y -r 2 -s 1627x256 -f image2 -i {input_dir}/%06d.png -pix_fmt yuv420p -vf \"pad=ceil(iw/2)*2:ceil(ih/2)*2\" {save_path}'
                os.system(cmd)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_dir', type=str, required=True)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)
