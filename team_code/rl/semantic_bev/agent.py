import os, yaml, json, pickle

from leaderboard.autoagents import autonomous_agent
from leaderboard.envs.sensor_interface import SensorInterface

from team_code.common.utils import mkdir_if_not_exists, parse_config
from team_code.rl.common.null_env import NullEnv
from team_code.rl.common.viz_utils import draw_text
from team_code.lbc.carla_project.src.common import CONVERTER, COLOR
from stable_baselines.sac.policies import MlpPolicy
from stable_baselines import SAC

from carla import VehicleControl

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

RESTORE = int(os.environ.get("RESTORE", 0))

def get_entry_point():
    return 'WaypointAgent'

class WaypointAgent(autonomous_agent.AutonomousAgent):
    def setup(self, path_to_conf_file=None):
        config = parse_config(path_to_conf_file)
        self.config = config.sac
        self.save_root = config.save_root
        self.track = autonomous_agent.Track.SENSORS

        # setup model and episode counter
        if RESTORE:
            self.restore()
        else:
            self.episode_num = -1
            self.obs_dim = self.config.waypoint_state_dim + 4
            self.action_dim = 2
            self.model = SAC(MlpPolicy, NullEnv(self.obs_dim, self.action_dim))

        self.save_images = self.config.save_images
        self.save_images_path  = f'{self.save_root}/images/episode_{self.episode_num:06d}'
        self.save_images_interval = 4

        self.cached_state = None
        self.cached_control = None
        self.cached_rinfo = 0
        self.cached_bev = None
        self.step = 0

    def restore(self):
        with open(f'{self.save_root}/logs/log.json', 'r') as f:
            log = json.load(f)
            self.episode_num = log['checkpoints'][-1]['index']
            print(f'restoring at episode {self.episode_num + 1}')
        weight_names = sorted(os.listdir(f'{self.save_root}/weights'))
        print(f'restoring model from {weight_names[-1]}')
        weight_path = f'{self.save_root}/weights/{weight_names[-1]}'
        self.model = SAC.load(weight_path)

        with open(f'{self.save_root}/logs/replay_buffer.pkl', 'rb') as f:
            self.model.replay_buffer = pickle.load(f)

    def sensors(self):
        return [
                    {
                    'type': 'sensor.camera.semantic_segmentation',
                    'x': 0.0, 'y': 0.0, 'z': 50.0,
                    'roll': 0.0, 'pitch': -90.0, 'yaw': 0.0,
                    'width': 384, 'height': 384, 'fov': 5 * 10.0,
                    'id': 'map'
                    },

                ]

    def destroy(self):
        if self.config.mode == 'train':
            self.sensor_interface = SensorInterface()

    def reset(self):
        self.step = 0
        self.cached_control = None
        self.cached_rinfo = 0
        self.episode_num += 1
        self.save_images_path  = f'{self.save_root}/images/episode_{self.episode_num:06d}'
        if self.config.save_images:
            mkdir_if_not_exists(self.save_images_path)

    def predict(self, state, burn_in=False):

        # compute controls
        if burn_in and not RESTORE:
            action = np.random.uniform(-1, 1, size=self.action_dim)
        else:
            action, _states = self.model.predict(state)

        #throttle, steer, brake = action
        throttle, steer = action
        throttle = float(throttle/2 + 0.5)
        steer = float(steer)
        #brake = float(brake/2 + 0.5)
        brake = False

        self.cached_state = state
        self.cached_control = VehicleControl(throttle, steer, brake)
        return action

    def run_step(self, input_data, timestamp):
        
        #self.cached_bev = input_data['bev'][1][:,:,:3]
        self.cached_map = COLOR[CONVERTER[input_data['map'][1][:,:,2]]]

        control = VehicleControl()
        if self.config.mode == 'train': # use cached training prediction           
            if self.cached_control:
                control = self.cached_control
        else: 
            # predict the action
            pass
        self.step += 1 
        return control

    def make_visualization(self, obs_norm):
        smap = np.array(self.cached_map)
        cv2.imshow('smap', smap)
        cv2.waitKey(1)

        if self.save_images:
            frame = self.step // self.save_images_interval
            save_path = f'{self.save_images_path}/{frame:06d}.png'
            cv2.imwrite(save_path, smap)
