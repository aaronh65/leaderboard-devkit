import os, yaml, json, pickle

from leaderboard.autoagents import autonomous_agent
from leaderboard.envs.sensor_interface import SensorInterface

from team_code.common.utils import mkdir_if_not_exists, parse_config
from team_code.rl.null_env import NullEnv
from team_code.rl.viz_utils import draw_text
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
            print('restoring')
            self.restore()
        else:
            self.model = SAC(MlpPolicy, NullEnv(6,3))
            self.episode_num = -1 # the first reset changes this to 0
        self.save_images_path  = f'{self.save_root}/images/episode_{self.episode_num:06d}'
        self.save_images_interval = 4

        self.cached_control = None
        self.cached_rinfo = 0
        self.cached_image = None
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
                    'type': 'sensor.camera.rgb',
                    'x': 0.0, 'y': 0.0, 'z': 25,
                    'roll': 0.0, 'pitch': -90.0, 'yaw': 0.0,
                    'width': 384, 'height': 384, 'fov': 75,
                    #'width': 256, 'height': 256, 'fov': 75,
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
        self.cached_control = None
        self.cached_rinfo = 0
        self.cached_image = None
        self.episode_num += 1
        self.save_images_path  = f'{self.save_root}/images/episode_{self.episode_num:06d}'
        if self.config.save_images:
            mkdir_if_not_exists(self.save_images_path)

    #def predict(self, state, model, burn_in=False):
    def predict(self, state, burn_in=False):

        # compute controls
        if burn_in and not RESTORE:
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
        
        # make the last timestep's visualization
        # now that we know what the reward was
        #if self.step > 0:
        #    self.make_visualization(input_data)

        self.cached_image = input_data['bev'][1][:,:,:3]

        control = VehicleControl()
        if self.config.mode == 'train': # use cached training prediction           
            if self.cached_control:
                control = self.cached_control
        else: 
            # predict the action
            pass
        self.step += 1 
        return control

    def make_visualization(self):
        image = np.array(self.cached_image)

        rinfo = self.cached_rinfo
        reward = rinfo['reward']
        rewdst = rinfo['dist_reward']
        rewyaw = rinfo['yaw_reward']
        rewvel = rinfo['vel_reward']

        text_strs = [
                f'Steer: {self.cached_control.steer:.3f}',
                f'Throttle: {self.cached_control.throttle:.3f}',
                f'Brake: {self.cached_control.brake:.3f}',
                f'Reward: {reward:.3f}',
                f'RewDst: {rewdst:.3f}',
                f'RewYaw: {rewyaw:.3f}',
                f'RewVel: {rewvel:.3f}',]

        for i, text in enumerate(text_strs):
            draw_text(image, text, (5, 20*(i+1)))

        cv2.imshow('debug', image)
        cv2.waitKey(1)

        if self.step % self.save_images_interval == 0:
            frame = self.step // self.save_images_interval
            save_path = f'{self.save_images_path}/{frame:06d}.png'
            cv2.imwrite(save_path, image)


