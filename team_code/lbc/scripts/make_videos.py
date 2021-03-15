import cv2
import os, argparse
from pathlib import Path

def main(args):

    videos = Path(f'{args.target_dir}/videos')
    routes = Path(f'{args.target_dir}/data').glob('*')

    width, height = None, None
    for route in sorted(list(routes)):
        route_reps = sorted(list(route.glob('*')))
        for dir_name in route_reps:
            
            # get save_path
            route, repetition = dir_name.parts[-2:]
            video_dir = videos / route
            video_dir.mkdir(parents=True, exist_ok=True)

            save_path = str(video_dir / f'{repetition}.mp4')
            debug_path = dir_name / 'debug'

            if width is None or height is None:
                fname = sorted(list(debug_path.glob('*')))[0]
                im = cv2.imread(str(fname))
                height, width, channels = im.shape

            if width % 2 == 1 or height % 2 == 1:
                cmd = f'ffmpeg -y -r 2 -s {width}x{height} -f image2 -i {debug_path}/%06d.png -pix_fmt yuv420p -vf \"pad=ceil(iw/2)*2:ceil(ih/2)*2\" {save_path}'
            else:
                cmd = f'ffmpeg -y -r 2 -s {width}x{height} -f image2 -i {debug_path}/%06d.png -pix_fmt yuv420p {save_path}'

            # make video
            #cmd = f'ffmpeg -y -r 2 -s 1627x256 -f image2 -i {input_dir}/%06d.png -pix_fmt yuv420p -vf \"pad=ceil(iw/2)*2:ceil(ih/2)*2\" {save_path}'
            os.system(cmd)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_dir', type=str, required=True)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)
