import os, yaml, json, pickle

from leaderboard.autoagents import autonomous_agent
from leaderboard.envs.sensor_interface import SensorInterface

from team_code.common.utils import mkdir_if_not_exists, parse_config
from team_code.rl.common.null_env import NullEnv
from team_code.rl.common.viz_utils import draw_text
from team_code.rl.common.semantic_utils import CONVERTER, COLOR
from stable_baselines.sac.policies import MlpPolicy
from stable_baselines.sac.policies import CnnPolicy
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
        self.obs_dim = (256,256,3,)
        self.action_dim = (2,)

        # setup model and episode counter
        if RESTORE:
            self.restore()
        else:
            self.episode_num = -1
            print('CREATING MODEL')
            self.model = SAC(CnnPolicy, NullEnv(self.obs_dim, self.action_dim, odtype=np.uint8, adtype=np.float32), buffer_size=1000, batch_size=16)
            print('CREATED MODEL')

        self.save_images = self.config.save_images
        self.save_images_path  = f'{self.save_root}/images/episode_{self.episode_num:06d}'
        self.save_images_interval = 4

        self.burn_in = False

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
                    'width': 256, 'height': 256, 'fov': 5 * 10.0,
                    'id': 'map'
                    },
                ]

    def destroy(self):
        if self.config.mode == 'train':
            self.sensor_interface = SensorInterface()

    def set_burn_in(self, burn_in):
        self.burn_in = burn_in

    def reset(self):

        self.episode_num += 1

        self.step = 0
        self.cached_map = np.zeros(self.obs_dim)
        self.cached_action = np.zeros(self.action_dim)
        self.cached_prev_map = np.zeros(self.obs_dim)
        self.cached_prev_action = np.zeros(self.action_dim)
        self.cached_rinfo = {'reward': -10}
        self.cached_done = False

        self.save_images_path  = f'{self.save_root}/images/episode_{self.episode_num:06d}'
        if self.config.save_images:
            mkdir_if_not_exists(self.save_images_path)

    def run_step(self, input_data, timestamp):

        # new state
        state = COLOR[CONVERTER[input_data['map'][1][:,:,2]]]
        
        # compute controls
        if self.burn_in and not RESTORE:
            action = np.random.uniform(-1, 1, size=self.action_dim)
        else:
            action, _states = self.model.predict(state)
        
        # add PID controller step?
        throttle, steer = action
        throttle = float(throttle/2 + 0.5)
        steer = float(steer)
        brake = False
        control = VehicleControl(throttle, steer, brake)

        # record things
        self.cached_prev_map = self.cached_map.copy()
        self.cached_prev_action = self.cached_action.copy()
        self.cached_map = state
        self.cached_action = action

        self.step += 1 
        return control

    def make_visualization(self):
        smap = np.array(self.cached_map)
        prev_smap = np.array(self.cached_prev_map)
        #combined = np.hstack([prev_smap, smap])
        combined = prev_smap

        throttle, steer = self.cached_prev_action
        throttle = float(throttle/2 + 0.5)
        steer = float(steer)

        rinfo = self.cached_rinfo
        reward = rinfo['reward']
        rewdst = rinfo['dist_reward']
        rewvel = rinfo['vel_reward']
        rewcmp = rinfo['route_reward']


        text_strs = [
                f'Steer: {steer:.3f}',
                f'Throttle: {throttle:.3f}',
                f'Reward: {reward:.3f}',
                f'RewDst: {rewdst:.3f}',
                f'RewVel: {rewvel:.3f}',
                f'RewCmp: {rewcmp:.3f}',]

        for i, text in enumerate(text_strs):
            draw_text(combined, text, (5, 20*(i+1)))


        cv2.imshow('combined', combined)
        cv2.waitKey(1)

        if self.save_images:
            frame = self.step // self.save_images_interval
            save_path = f'{self.save_images_path}/{frame:06d}.png'
            cv2.imwrite(save_path, combined)
