import time
import yaml
import cv2
import carla

from leaderboard.autoagents import autonomous_agent
from planner import RoutePlanner
from common.utils import *


class BaseAgent(autonomous_agent.AutonomousAgent):
    def setup(self, path_to_conf_file):
        self.track = autonomous_agent.Track.SENSORS
        config_type = type(path_to_conf_file)
        if config_type == str: # figure this part out
            self.config_path = path_to_conf_file
            with open(self.config_path, 'r') as f:
                config = yaml.load(f, Loader=yaml.Loader)
            self.config = dict_to_sns(config)
            #self.config.env = dict_to_sns(config['env'])
            #self.config.agent = dict_to_sns(config['agent'])
            #self.aconfig = self.config.agent
        else:
            self.config = path_to_conf_file
            self.aconfig = self.config.agent

        self.step = -1
        self.wall_start = time.time()
        self.initialized = False

    def _init(self):
        self.step = -1
        self._command_planner = RoutePlanner(7.5, 25.0, 256)
        self._command_planner.set_route(self._global_plan, True)
        self.initialized = True

    def _get_position(self, tick_data):
        gps = tick_data['gps']
        gps = (gps - self._command_planner.mean) * self._command_planner.scale

        return gps

    def sensors(self):
        return [
                {
                    'type': 'sensor.camera.rgb',
                    'x': 1.3, 'y': 0.0, 'z': 1.3,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'width': 256, 'height': 144, 'fov': 90,
                    'id': 'rgb'
                    },
                {
                    'type': 'sensor.camera.rgb',
                    'x': 1.2, 'y': -0.25, 'z': 1.3,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': -45.0,
                    'width': 256, 'height': 144, 'fov': 90,
                    'id': 'rgb_left'
                    },
                {
                    'type': 'sensor.camera.rgb',
                    'x': 1.2, 'y': 0.25, 'z': 1.3,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 45.0,
                    'width': 256, 'height': 144, 'fov': 90,
                    'id': 'rgb_right'
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
                {
                    'type': 'sensor.speedometer',
                    'reading_frequency': 20,
                    'id': 'speed'
                    }
                ]

    def tick(self, input_data, no_rgb=False):
        self.step += 1

        result = {}
        for key in input_data.keys():
            if 'rgb' in key:
                result[key] = cv2.cvtColor(input_data[key][1][:,:,:3], cv2.COLOR_BGR2RGB)

        result['gps'] = input_data['gps'][1][:2]
        result['speed'] = input_data['speed'][1]['speed']
        result['compass'] = input_data['imu'][1][-1]

        return result
