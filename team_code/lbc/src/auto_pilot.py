import os
import time
import datetime
import pathlib

import numpy as np
import cv2
import carla
import torch

from PIL import Image, ImageDraw

from lbc.carla_project.src.common import CONVERTER, COLOR
from lbc.carla_project.src.converter import Converter
from lbc.src.map_agent import MapAgent
from lbc.src.pid_controller import PIDController
from pathlib import Path

from leaderboard.envs.sensor_interface import SensorInterface

#HAS_DISPLAY = True
HAS_DISPLAY = int(os.environ.get('HAS_DISPLAY', 0))
ROUTE_NAME = os.environ.get('ROUTE_NAME', 0)
DEBUG = False
WEATHERS = [
        carla.WeatherParameters.ClearNoon,
        carla.WeatherParameters.ClearSunset,

        carla.WeatherParameters.CloudyNoon,
        carla.WeatherParameters.CloudySunset,

        carla.WeatherParameters.WetNoon,
        carla.WeatherParameters.WetSunset,

        carla.WeatherParameters.MidRainyNoon,
        carla.WeatherParameters.MidRainSunset,

        carla.WeatherParameters.WetCloudyNoon,
        carla.WeatherParameters.WetCloudySunset,

        carla.WeatherParameters.HardRainNoon,
        carla.WeatherParameters.HardRainSunset,

        carla.WeatherParameters.SoftRainNoon,
        carla.WeatherParameters.SoftRainSunset,
]


def get_entry_point():
    return 'AutoPilot'


def _numpy(carla_vector, normalize=False):
    result = np.float32([carla_vector.x, carla_vector.y])

    if normalize:
        return result / (np.linalg.norm(result) + 1e-4)

    return result


def _location(x, y, z):
    return carla.Location(x=float(x), y=float(y), z=float(z))


def _orientation(yaw):
    return np.float32([np.cos(np.radians(yaw)), np.sin(np.radians(yaw))])


def get_collision(p1, v1, p2, v2):
    A = np.stack([v1, -v2], 1)
    b = p2 - p1

    if abs(np.linalg.det(A)) < 1e-3:
        return False, None

    x = np.linalg.solve(A, b)
    collides = all(x >= 0) and all(x <= 1)

    return collides, p1 + x[0] * v1


class AutoPilot(MapAgent):
    def setup(self, path_to_conf_file):
        super().setup(path_to_conf_file)
        
        self.save_path = None
        self.save_freq = 1
        self.converter = Converter()
        
        # if block is untested
       
    def sensors(self):
        result = super().sensors()
        result = result[0:1] + result[3:]
        return result

    def _init(self):
        super()._init()

        self._turn_controller = PIDController(K_P=1.25, K_I=0.75, K_D=0.3, n=40)
        self._speed_controller = PIDController(K_P=5.0, K_I=0.5, K_D=1.0, n=40)

        if self.config.save_data or self.config.save_debug:

            route_name = os.environ['ROUTE_NAME']
            repetition = os.environ['REPETITION']
            self.save_path = Path(f'{self.config.save_root}/data/{route_name}/{repetition}')
            self.save_path.mkdir(parents=True, exist_ok=False)

        if self.config.save_debug:
            (self.save_path / 'debug').mkdir()

        if self.config.save_data:
            (self.save_path / 'rgb').mkdir()
            (self.save_path / 'rgb_left').mkdir()
            (self.save_path / 'rgb_right').mkdir()
            (self.save_path / 'topdown').mkdir()
            (self.save_path / 'measurements').mkdir()

    def reset(self):
        self.initialized = False
    def destroy(self):
        self.sensor_interface = SensorInterface()

    def _get_angle_to(self, pos, theta, target):
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)],
            ])

        aim = R.T.dot(target - pos)
        angle = -np.degrees(np.arctan2(-aim[1], aim[0]))
        angle = 0.0 if np.isnan(angle) else angle 

        return angle

    def _get_target(self, tick_data, cmd_route):
        theta = tick_data['theta']
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)],
            ])
        gps = self._get_position(tick_data) # method returns position in meters
        nodes = np.array([node for node, _ in cmd_route]) # (N,2)
        nodes = nodes - gps # center at agent position and rotate
        nodes = R.T.dot(nodes.T) # (2,2) x (2,N) = (2,N)
        nodes = nodes.T * 5.5 # (N,2) # to map frame (5.5 pixels per meter)
        nodes += [128,256]
        target = np.clip(nodes[1], 0, 256)
        return target

    def _get_control(self, target, far_target, tick_data):
        pos = self._get_position(tick_data)
        theta = tick_data['compass']
        speed = tick_data['speed']

        # Steering.
        angle_unnorm = self._get_angle_to(pos, theta, target)
        angle = angle_unnorm / 90

        steer = self._turn_controller.step(angle)
        steer = np.clip(steer, -1.0, 1.0)
        steer = round(steer, 3)

        # Acceleration.
        angle_far_unnorm = self._get_angle_to(pos, theta, far_target)
        should_slow = abs(angle_far_unnorm) > 45.0 or abs(angle_unnorm) > 5.0
        target_speed = 4 if should_slow else 7.0

        brake = self._should_brake()
        target_speed = target_speed if not brake else 0.0

        delta = np.clip(target_speed - speed, 0.0, 0.25)
        throttle = self._speed_controller.step(delta)
        throttle = np.clip(throttle, 0.0, 0.75)

        if brake:
            steer *= 0.5
            throttle = 0.0

        return steer, throttle, brake, target_speed

    def tick(self, input_data):
       data = super().tick(input_data)
       theta = data['compass']
       theta = 0.0 if np.isnan(theta) else theta
       theta = theta + np.pi / 2
       data['theta'] = theta
       return data

    def run_step(self, input_data, timestamp, noviz=False):
        if not self.initialized:
            self._init()

        if self.step % 100 == 0:
            index = (self.step // 100) % len(WEATHERS)
            self._world.set_weather(WEATHERS[index])

        data = self.tick(input_data)
        gps = self._get_position(data)


        wpt_route = self._waypoint_planner.run_step(gps)
        cmd_route = self._command_planner.run_step(gps)
        cmd_nodes = np.array([node for node, _ in cmd_route])
        cmd_cmds = [cmd for _, cmd in cmd_route]

        near_node, near_command = wpt_route[1]
        far_node, far_command = cmd_route[1]
        steer, throttle, brake, target_speed = self._get_control(near_node, far_node, data)

        data['target'] = self._get_target(data, cmd_route)

        control = carla.VehicleControl()
        control.steer = steer + 1e-2 * np.random.randn()
        control.throttle = throttle
        control.brake = float(brake)

        if self.config.save_data and self.step % self.save_freq == 0:
            self.save(far_node, near_command, steer, throttle, brake, target_speed, data)

        if (HAS_DISPLAY or self.config.save_debug) and not noviz:
            self.debug_display(data, steer, throttle, brake, target_speed, cmd_cmds, cmd_nodes, gps)
                
        
        
        return control

    def save(self, far_node, near_command, steer, throttle, brake, target_speed, tick_data):
        frame = self.step // self.save_freq

        Image.fromarray(tick_data['topdown']).save(
                self.save_path / 'topdown' / ('%06d.png' % frame))

        pos = self._get_position(tick_data)
        x, y = tick_data['target']
        theta = tick_data['theta']
        speed = tick_data['speed']

        data = {
            'x_target': x,
            'y_target': y,
            'x_position': pos[0],
            'y_position': pos[1],
            'x_command': far_node[0],
            'y_command': far_node[1],
            'command': near_command.value,
            'theta': theta,
            'speed': speed,
            'target_speed': target_speed,
            'steer': steer,
            'throttle': throttle,
            'brake': brake,
        }
        self.measurements = data

        (self.save_path / 'measurements' / ('%06d.json' % frame)).write_text(str(data))


    def debug_display(self, data, steer, throttle, brake, target_speed, cmd_cmds, cmd_nodes, gps, r=2):
        route_colors = [(255,255,255), (112,128,144), (47,79,79), (47,79,79)] 

        topdown = data['topdown']
        _topdown = Image.fromarray(COLOR[CONVERTER[topdown]])
        _topdown_draw = ImageDraw.Draw(_topdown)
        _rgb = Image.fromarray(data['rgb'])
        _rgb_draw = ImageDraw.Draw(_rgb)

        theta = data['theta']
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]])
        
        # route waypoints in map view
        route = cmd_nodes - gps
        route = R.T.dot(route.T).T
        route = route * 5.5
        route_map = route.copy() + [256, 256]
        route_map = route_map[:3]
        for i, (x,y) in enumerate(route_map):
            _topdown_draw.ellipse((x-2*r, y-2*r, x+2*r, y+2*r), route_colors[i])

        # route waypoints in cam view
        route_cam = route.copy()[:3].squeeze()
        route_cam = route_cam + [128, 256]
        route_cam = np.clip(route_cam, 0, 256)
        route_cam = self.converter.map_to_cam(torch.Tensor(route_cam)).numpy()
        for i, (x,y) in enumerate(route_cam[:3]):
            if i == 0:
                xt, yt = route_map[0]
                if xt < 128 or xt > 128+256 or yt < 5 or yt > 250: 
                    continue
            if x < 5 or x > 250 or y < 5 or y > 140: 
                continue
            _rgb_draw.ellipse((x-r, y-r, x+r, y+r), route_colors[i])

        #rgb = np.hstack((data['rgb_left'], data['rgb'], data['rgb_right']))
        #_rgb = Image.fromarray(np.hstack((data['rgb_left'], _rgb, data['rgb_right'])))

        _rgb = _rgb.resize((int(256 / _rgb.size[1] * _rgb.size[0]), 256))
        _draw = ImageDraw.Draw(_rgb)
        text_color = (255,255,255)
        _draw.text((5, 10), 'Steer: %.3f' % steer, text_color)
        _draw.text((5, 30), 'Throttle: %.3f' % throttle, text_color)
        _draw.text((5, 50), 'Brake: %s' % brake, text_color)
        _draw.text((5, 70), 'Speed: %.3f' % data['speed'], text_color)
        _draw.text((5, 90), 'Target: %.3f' % target_speed, text_color)
        cur_command, next_command = cmd_cmds[:2]
        _draw.text((5, 110), f'Current: {cur_command}', text_color)
        _draw.text((5, 130), f'Next: {next_command}', text_color)
        

        # (256, 144) -> (256/144*256, 256)
        _topdown = _topdown.resize((256,256))
        _combined = Image.fromarray(np.hstack((_rgb, _topdown)))

        #if self.step % 4 == 0 and self.config.save_data:
        #    _save_img = cv2.cvtColor(np.array(_combined), cv2.COLOR_BGR2RGB)
        #    frame_number = self.step // 4
        #    cv2.imwrite(str(self.save_path / 'debug' / f'{frame_number:06d}.png'), _save_img)

        if HAS_DISPLAY:
            cv2.imshow('map', cv2.cvtColor(np.array(_combined), cv2.COLOR_BGR2RGB))
            cv2.waitKey(1)


    def _should_brake(self):
        actors = self._world.get_actors()

        vehicle = self._is_vehicle_hazard(actors.filter('*vehicle*'))
        light = self._is_light_red(actors.filter('*traffic_light*'))
        walker = self._is_walker_hazard(actors.filter('*walker*'))

        return any(x is not None for x in [vehicle, light, walker])

    def _draw_line(self, p, v, z, color=(255, 0, 0)):
        if not DEBUG:
            return

        p1 = _location(p[0], p[1], z)
        p2 = _location(p[0]+v[0], p[1]+v[1], z)
        color = carla.Color(*color)

        self._world.debug.draw_line(p1, p2, 0.25, color, 0.01)

    def _is_light_red(self, lights_list):
        if self._vehicle.get_traffic_light_state() != carla.libcarla.TrafficLightState.Green:
            affecting = self._vehicle.get_traffic_light()

            for light in self._traffic_lights:
                if light.id == affecting.id:
                    return affecting

        return None

    def _is_walker_hazard(self, walkers_list):
        z = self._vehicle.get_location().z
        p1 = _numpy(self._vehicle.get_location())
        v1 = 10.0 * _orientation(self._vehicle.get_transform().rotation.yaw)

        self._draw_line(p1, v1, z+2.5, (0, 0, 255))

        for walker in walkers_list:
            v2_hat = _orientation(walker.get_transform().rotation.yaw)
            s2 = np.linalg.norm(_numpy(walker.get_velocity()))

            if s2 < 0.05:
                v2_hat *= s2

            p2 = -3.0 * v2_hat + _numpy(walker.get_location())
            v2 = 8.0 * v2_hat

            self._draw_line(p2, v2, z+2.5)

            collides, collision_point = get_collision(p1, v1, p2, v2)

            if collides:
                return walker

        return None

    def _is_vehicle_hazard(self, vehicle_list):
        z = self._vehicle.get_location().z

        o1 = _orientation(self._vehicle.get_transform().rotation.yaw)
        p1 = _numpy(self._vehicle.get_location())
        s1 = max(7.5, 2.0 * np.linalg.norm(_numpy(self._vehicle.get_velocity())))
        v1_hat = o1
        v1 = s1 * v1_hat

        self._draw_line(p1, v1, z+2.5, (255, 0, 0))

        for target_vehicle in vehicle_list:
            if target_vehicle.id == self._vehicle.id:
                continue

            o2 = _orientation(target_vehicle.get_transform().rotation.yaw)
            p2 = _numpy(target_vehicle.get_location())
            s2 = max(5.0, 2.0 * np.linalg.norm(_numpy(target_vehicle.get_velocity())))
            v2_hat = o2
            v2 = s2 * v2_hat

            p2_p1 = p2 - p1
            distance = np.linalg.norm(p2_p1)
            p2_p1_hat = p2_p1 / (distance + 1e-4)

            self._draw_line(p2, v2, z+2.5, (255, 0, 0))

            angle_to_car = np.degrees(np.arccos(v1_hat.dot(p2_p1_hat)))
            angle_between_heading = np.degrees(np.arccos(o1.dot(o2)))

            if angle_between_heading > 60.0 and not (angle_to_car < 15 and distance < s1):
                continue
            elif angle_to_car > 30.0:
                continue
            elif distance > s1:
                continue

            return target_vehicle

        return None
