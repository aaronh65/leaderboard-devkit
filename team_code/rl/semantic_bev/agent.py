import os, yaml, json, pickle
from collections import deque
from itertools import islice

from leaderboard.autoagents import autonomous_agent
from leaderboard.envs.sensor_interface import SensorInterface

from team_code.common.utils import mkdir_if_not_exists, parse_config
from team_code.rl.common.null_env import NullEnv
from team_code.rl.common.viz_utils import draw_text
from team_code.rl.common.semantic_utils import CONVERTER, COLOR
#from stable_baselines.sac.policies import MlpPolicy
from stable_baselines.sac.policies import CnnPolicy
from stable_baselines import SAC

from carla import VehicleControl

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

RESTORE = int(os.environ.get("RESTORE", 0))
HAS_DISPLAY = int(os.environ.get("HAS_DISPLAY", 0))

def get_entry_point():
    return 'WaypointAgent'

class WaypointAgent(autonomous_agent.AutonomousAgent):
    def setup(self, path_to_conf_file=None):
        config = parse_config(path_to_conf_file)
        self.config = config.agent
        self.save_root = config.save_root
        self.track = autonomous_agent.Track.SENSORS
        
        # setup model
        self.obs_dim = (self.config.bev_size,self.config.bev_size,self.config.history_size,)
        self.act_dim = (2,)
        obs_spec = ('box', 0, 255, self.obs_dim, np.uint8)
        act_spec = ('box', -1, 1, self.act_dim, np.float32)

        if RESTORE:
            self.restore()
            self.model.set_env(NullEnv(obs_spec, act_spec))
        else:
            self.episode_num = -1
            print('CREATING MODEL')
            self.model = SAC(
                    CnnPolicy, 
                    NullEnv(obs_spec, act_spec), 
                    buffer_size=self.config.buffer_size, 
                    batch_size=self.config.batch_size)
            print('CREATED MODEL')

        self.tensorboard_root = f'{config.save_root}/logs/tensorboard'
        self.model.tensorboard_log = self.tensorboard_root

        self.save_images = self.config.save_images
        self.save_images_path  = f'{self.save_root}/images/episode_{self.episode_num:06d}'
        self.save_images_interval = 4

        self.burn_in = False
        self.deterministic = False

    def restore(self):
        with open(f'{self.save_root}/logs/log.json', 'r') as f:
            log = json.load(f)
            self.episode_num = log['checkpoints'][-1]['index']
            print(f'restoring at episode {self.episode_num + 1}')
        weight_names = sorted(os.listdir(f'{self.save_root}/weights'))
        print(f'restoring model from {weight_names[-1]}')
        weight_path = f'{self.save_root}/weights/{weight_names[-1]}'
        self.model = SAC.load(weight_path)

        #print(f'{self.save_root}/logs/replay_buffer.pkl')
        #with open(f'{self.save_root}/logs/replay_buffer.pkl', 'rb') as f:
        #    self.model.replay_buffer = pickle.load(f)

    def sensors(self):
        return [
                    {
                    'type': 'sensor.camera.semantic_segmentation',
                    'x': 0.0, 'y': 0.0, 'z': 100.0,
                    'roll': 0.0, 'pitch': -90.0, 'yaw': 0.0,
                    'width': self.config.bev_size, 'height': self.config.bev_size, 'fov': 5 * 10.0,
                    'id': 'map'
                    },
                ]

    def destroy(self):
        if self.config.mode == 'train':
            self.sensor_interface = SensorInterface()

    def set_burn_in(self, burn_in):
        self.burn_in = burn_in

    def set_deterministic(self, deterministic):
        self.deterministic = deterministic

    def reset(self):

        self.episode_num += 1

        self.step = 0
        self.cached_maps = deque()
        self.cached_action = np.zeros(self.act_dim)
        self.cached_prev_action = np.zeros(self.act_dim)
        self.cached_rinfo = {'reward': -10}
        self.cached_done = False

        self.save_images_path  = f'{self.save_root}/images/episode_{self.episode_num:06d}'
        if self.config.save_images:
            mkdir_if_not_exists(self.save_images_path)

    def run_step(self, input_data, timestamp):

        # new state
        #$state = COLOR[CONVERTER[input_data['map'][1][:,:,2]]]
        state = input_data['map'][1][:,:,2]
        if self.step == 0:
            for _ in range(self.config.history_size+1):
                self.cached_maps.append(state)
        else:
            self.cached_maps.pop()
            self.cached_maps.appendleft(state)
        
        # compute controls
        if self.burn_in and not RESTORE:
            action = np.random.uniform(-1, 1, size=self.act_dim)
        else:
            obs = np.stack(islice(self.cached_maps, 0, self.config.history_size), axis=2)
            action, _states = self.model.predict(obs, deterministic=self.deterministic)
        
        # add PID controller step?
        throttle, steer = action
        throttle = np.clip(throttle/2 + 0.5, 0.0, 1.0)
        steer = np.clip(steer, -1.0, 1.0)
        brake = False
        control = VehicleControl(float(throttle), float(steer), float(brake))

        # record things
        self.cached_prev_action = self.cached_action.copy()
        self.cached_action = action

        self.step += 1 
        return control

    def make_visualization(self):
        #smap = np.array(COLOR[CONVERTER[self.cached_maps[0]]])
        #prev_smap = np.array(COLOR[CONVERTER[self.cached_maps[1]]])
        #combined = np.hstack([prev_smap, smap])
        combined = self.cached_maps[1]
        #combined = np.hstack(islice(self.cached_maps, 0, self.config.history_size+1))
        combined = COLOR[CONVERTER[combined]]

        throttle, steer = self.cached_prev_action
        throttle = float(throttle/2 + 0.5)
        steer = float(steer)

        rinfo = self.cached_rinfo
        reward = rinfo['reward']
        rewdst = rinfo['dist_reward']
        rewvel = rinfo['vel_reward']
        rewyaw = rinfo['yaw_reward']
        rewcmp = rinfo['route_reward']

        text_strs = [
                f'Steer: {steer:.3f}',
                f'Throttle: {throttle:.3f}',
                f'Reward: {reward:.3f}',
                f'RewDst: {rewdst:.3f}',
                f'RewVel: {rewvel:.3f}',
                f'RewYaw: {rewyaw:.3f}',
                f'RewCmp: {rewcmp:.3f}',]

        for i, text in enumerate(text_strs):
            draw_text(combined, text, (5, 20*(i+1)))

        if HAS_DISPLAY:
            cv2.imshow('combined', combined)
            cv2.waitKey(1)

        if self.save_images:
            frame = self.step // self.save_images_interval
            save_path = f'{self.save_images_path}/{frame:06d}.png'
            cv2.imwrite(save_path, combined)
