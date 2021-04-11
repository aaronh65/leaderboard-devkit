import os
import numpy as np
import cv2
import pickle as pkl
import torch
import torchvision
import carla

from PIL import Image, ImageDraw
from pathlib import Path

from lbc.carla_project.src.map_model import MapModel, plot_weights
from lbc.carla_project.src.dataset import preprocess_semantic
from lbc.carla_project.src.converter import Converter
from lbc.carla_project.src.common import CONVERTER, COLOR
from lbc.src.map_agent import MapAgent
from lbc.src.pid_controller import PIDController

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

DEBUG = int(os.environ.get('HAS_DISPLAY', 0))
ROUTE_NAME = os.environ.get('ROUTE_NAME', 0)

def get_entry_point():
    return 'PrivilegedAgent'

class PrivilegedAgent(MapAgent):
    def setup(self, path_to_conf_file):
        super().setup(path_to_conf_file)
        self.converter = Converter()
        project_root = self.config.project_root
        weights_path = self.config.weights_path
        self.net = MapModel.load_from_checkpoint(str(weights_path))
        self.net.cuda()
        self.net.eval()

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
            #(self.save_path / 'rgb_left').mkdir()
            #(self.save_path / 'rgb_right').mkdir()
            (self.save_path / 'topdown').mkdir()
            (self.save_path / 'measurements').mkdir()

    def sensors(self):

        sensors = super().sensors() 
        
        # get rid of rgb left/right cameras for faster runs
        rgb, _, _, imu, gps, speed, topdown = sensors
        sensors = [rgb, imu, gps, speed, topdown]
        return sensors

    def tick(self, input_data):
        result = super().tick(input_data)
        #result['image'] = np.concatenate(tuple(result[x] for x in ['rgb', 'rgb_left', 'rgb_right']), -1)
        result['image'] = result['rgb']

        theta = result['compass'] # heading angle from forward axis
        theta = 0.0 if np.isnan(theta) else theta
        theta = theta + np.pi / 2
        result['theta'] = theta
        #print((theta * 180 / np.pi)%360)
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)],
            ])
        result['R'] = R
        gps = self._get_position(result) # method returns position in meters
        
        # transform route waypoints to overhead map view
        route = self._command_planner.run_step(gps) # oriented in world frame
        nodes = np.array([node for node, _ in route]) # (N,2)
        nodes = nodes - gps # center at agent position and rotate
        nodes = R.T.dot(nodes.T) # (2,2) x (2,N) = (2,N)
        nodes = nodes.T * 5.5 # (N,2) # to map frame (5.5 pixels per meter)
        nodes += [128,256]
        #nodes = np.clip(nodes, 0, 256)
        commands = [command for _, command in route]
        target = np.clip(nodes[1], 0, 256)

        # populate results
        result['num_waypoints'] = len(route)
        result['route_map'] = nodes
        result['commands'] = commands
        result['target'] = target

        return result

    @torch.no_grad()
    def run_step_using_learned_controller(self, input_data, timestamp):
        if not self.initialized:
            self._init()

        tick_data = self.tick(input_data)

        img = torchvision.transforms.functional.to_tensor(tick_data['image'])
        img = img[None].cuda()

        target = torch.from_numpy(tick_data['target'])
        target = target[None].cuda()

        #points, (target_cam, _) = self.net.forward(img, target)
        points, (target_cam, _) = self.net.forward(img, target)
        control = self.net.controller(points).cpu().squeeze()

        steer = control[0].item()
        desired_speed = control[1].item()
        speed = tick_data['speed']

        brake = desired_speed < 0.4 or (speed / desired_speed) > 1.1

        delta = np.clip(desired_speed - speed, 0.0, 0.25)
        throttle = self._speed_controller.step(delta)
        throttle = np.clip(throttle, 0.0, 0.75)
        throttle = throttle if not brake else 0.0

        control = carla.VehicleControl()
        control.steer = steer
        control.throttle = throttle
        control.brake = float(brake)

        if DEBUG:
            debug_display(
                    tick_data, target_cam.squeeze(), points.cpu().squeeze(),
                    steer, throttle, brake, desired_speed,
                    self.step)

        return control

    @torch.no_grad()
    def run_step(self, input_data, timestamp):
        if not self.initialized:
            self._init()

        #rclist = CarlaDataProvider.get_route_completion_list()
        #iflist = CarlaDataProvider.get_infraction_list()

        #print(f'step {self.step}')
        #print(f'iflist {iflist}')
        #print(f'rclist {rclist}')

        tick_data = self.tick(input_data)
        topdown = Image.fromarray(tick_data['topdown'])
        topdown = topdown.crop((128, 0, 128+256, 256))
        topdown = np.array(topdown)
        topdown = preprocess_semantic(topdown)
        topdown = topdown[None].cuda()

        target = torch.from_numpy(tick_data['target'])
        #print(tick_data['target'])
        target = target[None].cuda()

        points, weights, target_heatmap = self.net.forward(topdown, target) # world frame


        points_map = points.clone().cpu().squeeze()
        points_map = points_map + 1
        points_map = points_map / 2 * 256
        points_map = np.clip(points_map, 0, 256)
        points_world = self.converter.map_to_world(points_map).numpy()

        tick_data['points_map'] = points_map

        #img = tick_data['image']

        aim = (points_world[1] + points_world[0]) / 2.0
        angle = np.degrees(np.pi / 2 - np.arctan2(aim[1], aim[0])) / 90
        steer = self._turn_controller.step(angle)
        steer = np.clip(steer, -1.0, 1.0)
        desired_speed = np.linalg.norm(points_world[0] - points_world[1]) * 2.0
        tick_data['aim_world'] = aim

        speed = tick_data['speed']
        brake = desired_speed < 0.4 or (speed / desired_speed) > 1.1

        delta = np.clip(desired_speed - speed, 0.0, 0.25)
        throttle = self._speed_controller.step(delta)
        throttle = np.clip(throttle, 0.0, 0.75)
        throttle = throttle if not brake else 0.0

        control = carla.VehicleControl()
        control.steer = steer
        control.throttle = throttle
        control.brake = float(brake)
        #print(timestamp) # GAMETIME

        if DEBUG or self.config.save_debug:

            # transform image model cam points to overhead BEV image (spectator frame?)
            self.debug_display(
                    tick_data, steer, throttle, brake, desired_speed)
        if DEBUG:    
            images = plot_weights(topdown, target, points, weights)
            cv2.imshow('heatmaps', images[0])
        return control

    def debug_display(self, tick_data, steer, throttle, brake, desired_speed, r=2):

        text_color = (255,255,255)
        aim_color = (60,179,113) # dark green
        lbc_color = (178,34,34) # dark red
        route_colors = [(255,255,255), (112,128,144), (47,79,79), (47,79,79)] 

        topdown = tick_data['topdown']
        _topdown = Image.fromarray(COLOR[CONVERTER[topdown]])
        _topdown_draw = ImageDraw.Draw(_topdown)

        # model points
        points_map = tick_data['points_map']
        points_td = points_map.numpy() + [128, 0] # map view to full topdown view
        for i, (x,y) in enumerate(points_td):
            _topdown_draw.ellipse((x-2*r, y-2*r, x+2*r, y+2*r), (255,0,0))
        
        # control point
        aim_world = np.array(tick_data['aim_world'])
        aim_map = self.converter.world_to_map(torch.Tensor(aim_world)).numpy()
        x,y = aim_map + [128,0]
        _topdown_draw.ellipse((x-2, y-2, x+2, y+2), (0,255,0))

        # route waypoints
        route_map = tick_data['route_map']
        route_td = route_map + [128, 0]
        for i, (x, y) in enumerate(route_td[:3]):
            _topdown_draw.ellipse((x-2*r, y-2*r, x+2*r, y+2*r), route_colors[i])

        # make RGB images

        # draw center RGB image
        _rgb = Image.fromarray(tick_data['rgb'])
        _draw_rgb = ImageDraw.Draw(_rgb)
        points_cam = self.converter.map_to_cam(points_map).cpu()
        for x, y in points_cam: # image model waypoints
            _draw_rgb.ellipse((x-2, y-2, x+2, y+2), lbc_color)

        x,y = self.converter.world_to_cam(torch.Tensor(aim_world)).numpy()
        _draw_rgb.ellipse((x-2, y-2, x+2, y+2), aim_color)

        # draw route waypoints in RGB image
        route_cam = self.converter.map_to_cam(torch.Tensor(route_map)).numpy()
        for i, (x, y) in enumerate(route_cam[:3]):
            if i == 0: # waypoint we just passed
                xt, yt = route_map[0]
                if xt < 5 or xt > 250 or yt < 5 or yt > 250: continue
            if x < 5 or x > 250 or y < 5 or y > 140: continue
            _draw_rgb.ellipse((x-2*r, y-2*r, x+2*r, y+2*r), route_colors[i])

        #_combined = Image.fromarray(np.hstack([tick_data['rgb_left'], _rgb, tick_data['rgb_right']]))
        #_combined = Image.fromarray(_rgb)
        _combined = _rgb
        _combined = _combined.resize((int(256/ _combined.size[1] * _combined.size[0]), 256))
        _draw = ImageDraw.Draw(_combined)

        # draw debug text
        _draw.text((5, 10), 'Steer: %.3f' % steer, text_color)
        _draw.text((5, 30), 'Throttle: %.3f' % throttle, text_color)
        _draw.text((5, 50), 'Brake: %s' % brake, text_color)
        _draw.text((5, 70), 'Speed: %.3f' % tick_data['speed'], text_color)
        _draw.text((5, 90), 'Desired: %.3f' % desired_speed, text_color)
        cur_command, next_command = tick_data['commands'][:2]
        _draw.text((5, 110), f'Current: {cur_command}', text_color)
        _draw.text((5, 130), f'Next: {next_command}', text_color)

        _topdown = _topdown.resize((256, 256))
        _save_img = Image.fromarray(np.hstack([_combined, _topdown]))
        _save_img = cv2.cvtColor(np.array(_save_img), cv2.COLOR_BGR2RGB)
        if self.step % 10 == 0 and self.config.save_debug:
            frame_number = self.step // 10
            save_path = self.save_path / 'debug' / f'{frame_number:06d}.png'
            cv2.imwrite(str(save_path), _save_img)
        if DEBUG:
            cv2.imshow('debug', _save_img)
            cv2.waitKey(1)
 
