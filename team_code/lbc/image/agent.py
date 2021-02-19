import os 
import numpy as np
import cv2
#import json
import pickle as pkl
import torch
import torchvision
import carla

from PIL import Image, ImageDraw
from pathlib import Path

from lbc.carla_project.src.image_model import ImageModel
from lbc.carla_project.src.converter import Converter
from lbc.common.base_agent import BaseAgent
from lbc.common.pid_controller import PIDController

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

DEBUG = int(os.environ.get('HAS_DISPLAY', 0))
ROUTE_NAME = os.environ.get('ROUTE_NAME', 0)

def get_entry_point():
    return 'ImageAgent'

class ImageAgent(BaseAgent):
    def setup(self, path_to_conf_file):
        super().setup(path_to_conf_file)

        self.converter = Converter()
        project_root = self.config.project_root
        weights_path = self.config.weights_path
        self.net = ImageModel.load_from_checkpoint(f'{project_root}/{weights_path}')
        self.net.cuda()
        self.net.eval()

    def _init(self):
        super()._init()

        self._turn_controller = PIDController(K_P=1.25, K_I=0.75, K_D=0.3, n=40)
        self._speed_controller = PIDController(K_P=5.0, K_I=0.5, K_D=1.0, n=40)

        self.save_images_path = Path(f'{self.config.save_root}/images/{ROUTE_NAME}')
        self.save_image_dim=(1371,256)
        #self.save_path.mkdir()


    def tick(self, input_data):
        result = super().tick(input_data)
        result['image'] = np.concatenate(tuple(result[x] for x in ['rgb', 'rgb_left', 'rgb_right']), -1)

        theta = result['compass']
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
        nodes = np.clip(nodes, 0, 256)
        commands = [command for _, command in route]

        # populate results
        result['num_waypoints'] = len(route)
        result['route_map'] = nodes
        result['commands'] = commands
        result['target'] = nodes[1]

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

        # prepare image model inputs
        img = torchvision.transforms.functional.to_tensor(tick_data['image'])
        img = img[None].cuda()
        target = torch.from_numpy(tick_data['target'])
        target = target[None].cuda()

        # forward through NN and process output
        points, (target_cam, _) = self.net.forward(img, target)
        #heatmap = _.cpu().numpy()
        #heatmap = cv2.flip(heatmap[0][0], 0)
        #cv2.imshow('heatmap', heatmap)
        points_cam = points.clone().cpu()
        points_cam[..., 0] = (points_cam[..., 0] + 1) / 2 * img.shape[-1]
        points_cam[..., 1] = (points_cam[..., 1] + 1) / 2 * img.shape[-2]
        points_cam = points_cam.squeeze()
        points_world = self.converter.cam_to_world(points_cam).numpy() 
        tick_data['points_world'] = points_world

        # decide on control waypoint
        aim = (points_world[1] + points_world[0]) / 2.0
        tick_data['aim_world'] = aim

        # compute steer
        angle = np.degrees(np.pi / 2 - np.arctan2(aim[1], aim[0])) / 90
        steer = self._turn_controller.step(angle)
        steer = np.clip(steer, -1.0, 1.0)

        # compute throttle
        speed = tick_data['speed']
        desired_speed = np.linalg.norm(points_world[0] - points_world[1]) * 2.0 # m/s
        brake = desired_speed < 0.4 or (speed / desired_speed) > 1.1
        delta = np.clip(desired_speed - speed, 0.0, 0.25)
        throttle = self._speed_controller.step(delta)
        throttle = np.clip(throttle, 0.0, 0.75)
        throttle = throttle if not brake else 0.0

        # create control object
        control = carla.VehicleControl()
        control.steer = steer
        control.throttle = throttle
        control.brake = float(brake)
        if DEBUG or self.config.save_images:

            # make display image
            tick_data['points_cam'] = points.cpu().squeeze()
            tick_data['points_map'] = self.converter.cam_to_map(points_cam).numpy()
            self.debug_display(
                    tick_data, target_cam.squeeze(), 
                    steer, throttle, brake, desired_speed)

        return control

    def debug_display(self, tick_data, target_cam, steer, throttle, brake, desired_speed):

        # make BEV image

        # transform aim from world to map
        aim_world = np.array(tick_data['aim_world'])
        aim_map = self.converter.world_to_map(torch.Tensor(aim_world)).numpy()

        # append to image model points and plot
        points_plot = np.vstack([tick_data['points_map'], aim_map])
        points_plot = points_plot - [128,256] # center at origin
        points_plot = tick_data['R'].dot(points_plot.T).T
        points_plot = points_plot * -1 # why is this required?
        points_plot = points_plot + 256/2 # recenter origin in middle of plot
        _waypoint_img = self._command_planner.debug.img
        for x, y in points_plot:
            ImageDraw.Draw(_waypoint_img).ellipse((x-2, y-2, x+2, y+2), (0, 191, 255))
        x, y = points_plot[-1]
        ImageDraw.Draw(_waypoint_img).ellipse((x-2, y-2, x+2, y+2), (255, 105, 147))

        # make RGB images

        # draw center RGB image
        _rgb = Image.fromarray(tick_data['rgb'])
        _draw_rgb = ImageDraw.Draw(_rgb)
        for x, y in tick_data['points_cam']: # image model waypoints
            x = (x + 1)/2 * 256
            y = (y + 1)/2 * 144
            _draw_rgb.ellipse((x-2, y-2, x+2, y+2), (0, 191, 255))

        # transform aim from world to cam
        aim_world = np.array(tick_data['aim_world'])
        aim_cam = self.converter.world_to_cam(torch.Tensor(aim_world)).numpy()
        x, y = aim_cam
        _draw_rgb.ellipse((x-2, y-2, x+2, y+2), (255, 105, 147))

        # draw route waypoints in RGB image
        route_map = np.array(tick_data['route_map'])
        route_map = route_map[:3].squeeze()
        route_cam = self.converter.map_to_cam(torch.Tensor(route_map)).numpy()
        for i, (x, y) in enumerate(route_cam):
            if i == 0: # waypoint we just passed
                if y >= 139 or x <= 2 or x >= 254: # bottom of frame (behind us)
                    continue
                color = (0, 255, 0) # green 
            elif i == 1: # target
                color = (255, 0, 0) # red
            elif i == 2: # beyond target
                color = (139, 0, 139) # darkmagenta
            else:
                continue
            _draw_rgb.ellipse((x-2, y-2, x+2, y+2), color)

        _combined = Image.fromarray(np.hstack([tick_data['rgb_left'], _rgb, tick_data['rgb_right']]))
        _draw = ImageDraw.Draw(_combined)

        # draw debug text
        text_color = (139, 0, 139) #darkmagenta
        _draw.text((5, 10), 'Steer: %.3f' % steer, text_color)
        _draw.text((5, 30), 'Throttle: %.3f' % throttle, text_color)
        _draw.text((5, 50), 'Brake: %s' % brake, text_color)
        _draw.text((5, 70), 'Speed: %.3f' % tick_data['speed'], text_color)
        _draw.text((5, 90), 'Desired: %.3f' % desired_speed, text_color)
        cur_command, next_command = tick_data['commands'][:2]
        _draw.text((5, 110), f'Current: {cur_command}', text_color)
        _draw.text((5, 130), f'Next: {next_command}', text_color)

        # compose image to display/save
        _rgb_img = cv2.resize(np.array(_combined), self.save_image_dim, interpolation=cv2.INTER_AREA)
        _save_img = Image.fromarray(np.hstack([_rgb_img, _waypoint_img]))
        _save_img = cv2.cvtColor(np.array(_save_img), cv2.COLOR_BGR2RGB)

        if self.step % 10 == 0 and self.config.save_images:
            frame_number = self.step // 10
            rep_number = int(os.environ.get('REP',0))
            save_path = self.save_images_path / f'repetition_{rep_number:02d}' / f'{frame_number:06d}.png'
            cv2.imwrite(str(save_path), _save_img)

        if DEBUG:
            cv2.imshow('debug', _save_img)
            cv2.waitKey(1)
