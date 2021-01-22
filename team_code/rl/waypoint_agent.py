import os, yaml

#from team_code.base_agent import BaseAgent
from leaderboard.autoagents import autonomous_agent
from leaderboard.envs.sensor_interface import SensorInterface

from team_code.common.utils import *
from team_code.rl.null_env import NullEnv
from stable_baselines.sac.policies import MlpPolicy
from stable_baselines import SAC

from carla import VehicleControl

import cv2
import numpy as np


def get_entry_point():
    return 'WaypointAgent'

class WaypointAgent(autonomous_agent.AutonomousAgent):
    def setup(self, path_to_conf_file=None):
        config_type = type(path_to_conf_file)
        if config_type == str:
            self.config_path = path_to_conf_file
            with open(self.config_path, 'r') as f:
                config = yaml.load(f, Loader=yaml.Loader)
            self.config = Bunch(config)
        elif config_type == dict:
            self.config = Bunch(path_to_conf_file)
        elif config_type == Bunch:
            self.config = path_to_conf_file

        self.track = autonomous_agent.Track.SENSORS
        self.model = SAC(MlpPolicy, NullEnv(6,3))
        self.cached_control = None
        self.step = 0
        self.episode_num = -1 # the first reset changes this to 0
        self.save_images_path  = f'{self.config.save_root}/images/episode_{self.episode_num:06d}'
        self.save_images_interval = 5

    def sensors(self):
        return [
                {
                    'type': 'sensor.camera.rgb',
                    'x': 0.0, 'y': 0.0, 'z': 25,
                    'roll': 0.0, 'pitch': -90.0, 'yaw': 0.0,
                    'width': 384, 'height': 384, 'fov': 75,
                    'id': 'bev'
                    },
                 
                {
                    'type': 'sensor.other.imu',
                    'x': 0.0, 'y': 0.0, 'z': 0.0,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'sensor_tick': 0.05,
                    'id': 'imu'
                    },
                {
                    'type': 'sensor.other.gnss',
                    'x': 0.0, 'y': 0.0, 'z': 0.0,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'sensor_tick': 0.01,
                    'id': 'gps'
                    },
                ]

    def destroy(self):
        if self.config.mode == 'train':
            self.sensor_interface = SensorInterface()

    def reset(self):
        self.step = 0
        self.episode_num += 1
        self.save_images_path  = f'{self.config.save_root}/images/episode_{self.episode_num:06d}'
        if self.config.save_images:
            mkdir_if_not_exists(self.save_images_path)

    #def predict(self, state, model, burn_in=False):
    def predict(self, state, burn_in=False):

        # compute controls
        if burn_in:
            action = np.random.uniform(-1, 1, size=3)
        else:
            action, _states = self.model.predict(state)

        throttle, steer, brake = action
        throttle = float(throttle/2 + 0.5)
        steer = float(steer)
        #brake = float(brake/2 + 0.5)
        brake = False
        self.cached_control = VehicleControl(throttle, steer, brake)
        return action

    def run_step(self, input_data, timestamp):
        
        if self.config.save_images:
            image = input_data['bev'][1][:, :, :3] # what's the last number?
            cv2.imshow('debug', image)
            cv2.waitKey(1)
            if self.step % self.save_images_interval == 0:
                frame = self.step // self.save_images_interval
                save_path = f'{self.save_images_path}/{frame:06d}.png'
                cv2.imwrite(save_path, image)

        control = VehicleControl()
        if self.config.mode == 'train': # use cached training prediction           
            if self.cached_control:
                control = self.cached_control
        else: 
            # predict the action
            pass
        self.step += 1 
        return control

