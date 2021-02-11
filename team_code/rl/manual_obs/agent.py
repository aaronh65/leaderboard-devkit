import cv2
import numpy as np
import os, yaml, json, pickle
from PIL import Image, ImageDraw, ImageFont
from carla import VehicleControl

from leaderboard.autoagents import autonomous_agent
from leaderboard.envs.sensor_interface import SensorInterface
from team_code.common.utils import mkdir_if_not_exists, parse_config
from team_code.rl.common.null_env import NullEnv
from team_code.rl.common.viz_utils import draw_text
from team_code.lbc.src.pid_controller import PIDController
#from team_code.rl.common.semantic_utils import CONVERTER, COLOR
from stable_baselines.sac.policies import MlpPolicy
from stable_baselines import SAC



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
        self.tensorboard_root = f'{config.save_root}/logs/tensorboard'
        
        # setup model
        self.obs_dim = (self.config.waypoint_state_dim + 4,)
        #self.obs_dim = (self.config.waypoint_state_dim + 5,)
        self.act_dim = (2,)
        obs_spec = ('box', -1, 1, self.obs_dim, np.float32)
        act_spec = ('box', -1, 1, self.act_dim, np.float32)

        if RESTORE:
            self.restore()
            self.model.set_env(NullEnv(obs_spec, act_spec))
        else:
            self.model = SAC(MlpPolicy, NullEnv(obs_spec, act_spec))
            self.episode_num = -1 # the first reset changes this to 0
        self.model.tensorboard_log = self.tensorboard_root

        #print(self.model._vec_normalize_env)
        self.save_images = self.config.save_images
        self.save_images_path  = f'{self.save_root}/images/episode_{self.episode_num:06d}'
        self.save_images_interval = 4

        self._turn_controller = PIDController(K_P=1.25, K_I=0.75, K_D=0.3, n=40)
        self._speed_controller = PIDController(K_P=5.0, K_I=0.5, K_D=1.0, n=40)

    def restore(self):
        with open(f'{self.save_root}/logs/log.json', 'r') as f:
            log = json.load(f)
            self.episode_num = log['checkpoints'][-1]['index']
            print(f'restoring at episode {self.episode_num + 1}')
        weight_names = sorted(os.listdir(f'{self.save_root}/weights'))
        print(f'restoring model from {weight_names[-1]}')
        weight_path = f'{self.save_root}/weights/{weight_names[-1]}'
        self.model = SAC.load(weight_path)

        #with open(f'{self.save_root}/logs/replay_buffer.pkl', 'rb') as f:
        #    self.model.replay_buffer = pickle.load(f)

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
                #{
                #    'type': 'sensor.speedometer',
                #    'reading_frequency': 20,
                #    'id': 'speed'
                #    }

                ]

    def destroy(self):
        if self.config.mode == 'train':
            self.sensor_interface = SensorInterface()

    def reset(self):
        self.episode_num += 1
        self.cached_state = None
        self.cached_control = None
        self.cached_rinfo = 0
        self.cached_bev = None
        self.step = 0
        self.save_images_path  = f'{self.save_root}/images/episode_{self.episode_num:06d}'
        if self.config.save_images:
            mkdir_if_not_exists(self.save_images_path)

    def _get_control(self, action):

        if self.config.pid:

            #throttle, steer, brake = action
            next_speed, angle = action

            angle = angle * 180
            steer = self._turn_controller.step(angle / 90)
            steer = np.clip(steer, -1.0, 1.0)

            current_speed = (state[5]+1) * 40 # km/h
            current_speed = current_speed * 1000/3600 # m/s
            next_speed = (next_speed+1) * 40
            next_speed = next_speed * 1000/3600
            brake = next_speed < 0.4 or (current_speed / next_speed) > 1.1

            delta = np.clip(next_speed - current_speed, 0.0, 0.25)
            throttle = self._speed_controller.step(delta)

        else:

            throttle, steer = action
            #throttle = throttle/2 + 0.5
            #throttle = np.clip(throttle, 0, 1)
            #steer = np.clip(steer, -1, 1)
            #brake = float(brake/2 + 0.5)
            brake = False

        throttle = np.clip(throttle, 0.0, 1.0)
        throttle = throttle if not brake else 0.0
        steer = np.clip(steer, -1.0, 1.0)
        return VehicleControl(float(throttle), float(steer), float(brake))

    def predict(self, state, burn_in=False, deterministic=False):

        self.cached_state = state
        # compute controls
        if burn_in and not RESTORE:
            action = np.random.uniform(-1, 1, size=self.act_dim)
            self.cached_mean = action
            self.cached_std = np.zeros(self.act_dim)
        else:
            action, _states = self.model.predict(state, deterministic=deterministic)
            self.cached_mean, self.cached_std = self.model.policy_tf.proba_step([state])
            self.cached_mean = self.cached_mean.squeeze()
            self.cached_std = np.zeros(self.act_dim) if deterministic else self.cached_std.squeeze()
        action = np.clip(action, -1, 1)
        self.cached_control = self._get_control(action)
        return action

    def run_step(self, input_data, timestamp):
        
        self.cached_bev = input_data['bev'][1][:,:,:3]
        #self.cached_map = COLOR[CONVERTER[input_data['map'][1][:,:,2]]]

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
        #smap = np.array(self.cached_map)
        #cv2.imshow('smap', smap)

        bev = np.array(self.cached_bev)
        height, width = bev.shape[:2]

        rinfo = self.cached_rinfo
        reward = rinfo['reward']
        rewdst = rinfo['dist_reward']
        rewvel = rinfo['vel_reward']
        rewyaw = rinfo['yaw_reward']
        rewcmp = rinfo['route_reward']
        rewtra = rinfo['traffic_reward']

        #print(np.amax(np.abs(self.cached_state)))
        distance = (self.cached_state[0]+1) / 2 * obs_norm[0]
        heading = self.cached_state[1] * obs_norm[1]
        z = self.cached_state[2] * obs_norm[2]
        dyaw = self.cached_state[3] * obs_norm[3]
        curvature = self.cached_state[4] * 180
        #tl_dist = (self.cached_state[8]+1) * 12.5
        
        throt = self.cached_mean[0]/2 + 0.5
        throt_str = f'{throt:.2f},{self.cached_std[0]/2:.2f}'
        steer_str = f'{self.cached_mean[1]:.2f},{self.cached_std[1]:.2f}'
        left_text_strs = [
                f'Distance: {distance:.3f}', # add curvature after?
                f'Heading: {heading:.3f}',
                f'Height: {z:.3f}',
                f'YawDiff: {dyaw:.3f}',
                f'Curve: {curvature:.3f}',
                f'Throt: {throt_str}',
                f'Steer: {steer_str}',
                #f'TL dist: {tl_dist:.3f}',
                ]

        right_text_strs = [
                f'Reward: {reward:.3f}',
                f'RewDst: {rewdst:.3f}',
                f'RewVel: {rewvel:.3f}',
                f'RewYaw: {rewyaw:.3f}',
                f'RewCmp: {rewcmp:.3f}',
                #f'RewTra: {rewtra:.3f}'
                ]

        for i, text in enumerate(left_text_strs):
            draw_text(bev, text, (5, 20*(i+1)))
        for i, text in enumerate(right_text_strs):
            draw_text(bev, text, (width-130, 20*(i+1)))

        if HAS_DISPLAY:
            cv2.imshow('bev', bev)
            cv2.waitKey(1)

        if self.save_images and self.step % self.save_images_interval == 0:
            frame = self.step // self.save_images_interval
            save_path = f'{self.save_images_path}/{frame:06d}.png'
            cv2.imwrite(save_path, bev)

