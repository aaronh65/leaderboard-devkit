from PIL import Image
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt


CONVERTER = np.uint8([
    0,    # unlabeled
    0,    # building
    0,    # fence
    0,    # other
    1,    # ped
    0,    # pole
    2,    # road line
    3,    # road
    4,    # sidewalk
    0,    # vegetation
    5,    # car
    0,    # wall
    0,    # traffic sign
    0,    # sky
    0,    # ground
    0,    # bridge
    0,    # railtrack
    0,    # guardrail
    0,    # trafficlight
    0,    # static
    0,    # dynamic
    0,    # water
    0,    # terrain
    6,
    7,
    8,
    ])


COLOR = np.uint8([
        (  0,   0,   0),    # unlabeled
        (220,  20,  60),    # ped
        (157, 234,  50),    # road line
        (128,  64, 128),    # road
        (244,  35, 232),    # sidewalk
        (  0,   0, 142),    # car
        (255,   0,   0),    # red light
        (255, 255,   0),    # yellow light
        (  0, 255,   0),    # green light
        ])

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', type=str, required=True)
args = parser.parse_args()

path = Path(args.dataset_dir)
episodes = list(sorted(Path(path).glob('*')))[:-1]
measurements = list(sorted((episodes[-4] / 'measurements').glob('*.json')))
measurements = pd.DataFrame([eval(x.read_text()) for x in measurements])
# reward
rewards = measurements['reward'].to_numpy()
rewards = rewards[:50]
plt.plot(np.arange(rewards.shape[0]), rewards)
plt.show()

# images
topdowns = list(sorted((episodes[-1] / 'topdown').glob('*.png')))
if False:
    for topdown in topdowns:
        img = Image.open(topdown)
        img = COLOR[CONVERTER[np.array(img)]]
        cv2.imshow('topdown', img)
    cv2.waitKey(50)
