import os
import cv2
import argparse

def main(args):

    video_dir = os.path.join(args.target_dir, 'videos')
    image_dir = os.path.join(args.target_dir, 'images')
    assert os.path.exists(image_dir), 'no image directory'
    routes = sorted(os.listdir(image_dir))

    width, height = None, None
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

            if width is None or height is None:
                fname = os.listdir(input_dir)[0]
                impath = os.path.join(input_dir, fname)
                im = cv2.imread(impath)
                height, width, channels = im.shape

            if width % 2 == 1 or height % 2 == 1:
                cmd = f'ffmpeg -y -r 2 -s {width}x{height} -f image2 -i {input_dir}/%06d.png -pix_fmt yuv420p -vf \"pad=ceil(iw/2)*2:ceil(ih/2)*2\" {save_path}'
            else:
                cmd = f'ffmpeg -y -r 2 -s {width}x{height} -f image2 -i {input_dir}/%06d.png -pix_fmt yuv420p {save_path}'


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
