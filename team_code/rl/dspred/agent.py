import os
import carla
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision

from PIL import Image, ImageDraw
from pathlib import Path

#from lbc.carla_project.src.map_model import MapModel
from rl.dspred.map_model import MapModel, fuse_vmaps
from lbc.carla_project.src.dataset import preprocess_semantic
from lbc.carla_project.src.converter import Converter
from lbc.carla_project.src.common import CONVERTER, COLOR
from lbc.common.map_agent import MapAgent
from lbc.common.pid_controller import PIDController

from leaderboard.envs.sensor_interface import SensorInterface
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

HAS_DISPLAY = int(os.environ.get('HAS_DISPLAY', 0))

def get_entry_point():
    return 'DSPredAgent'

class DSPredAgent(MapAgent):
    def setup(self, path_to_conf_file):
        super().setup(path_to_conf_file)
        self.converter = Converter()

        project_root = self.config.project_root
        self.debug_path = None
        self.data_path = None

        weights_path = self.aconfig.weights_path
        self.net = MapModel.load_from_checkpoint(f'{project_root}/{weights_path}')
        self.net.cuda()
        self.net.eval()

        if self.aconfig.mode == 'dagger':
            self.expert = MapModel.load_from_checkpoint(f'{project_root}/{weights_path}')
            self.expert.cuda()
            self.expert.eval()

        self.burn_in = False

    def _init(self):
        super()._init()
        self._turn_controller = PIDController(K_P=1.25, K_I=0.75, K_D=0.3, n=40)
        self._speed_controller = PIDController(K_P=5.0, K_I=0.5, K_D=1.0, n=40)

    def reset(self, route_name, repetition):
        self.initialized = False

        # make debug directories
        save_root = self.config.save_root
        self.save_path = Path(f'{save_root}/{route_name}/{repetition}')
        self.save_path.mkdir(parents=True, exist_ok=True)

        print(self.save_path)

        if self.config.save_debug:
            (self.save_path / 'debug').mkdir()
        if self.config.save_data:
            (self.save_path / 'topdown').mkdir()
            (self.save_path / 'points_lbc').mkdir()
            (self.save_path / 'points_dqn').mkdir()
            (self.save_path / 'measurements').mkdir()

    def save_data(self,  tick_data):
        sp = self.save_path
        frame = f'{self.step:06d}'
        with open(sp / 'points_lbc' / f'{frame}.npy', 'wb') as f:
            np.save(f, tick_data['points_lbc'])
        with open(sp / 'points_dqn' / f'{frame}.npy', 'wb') as f:
            np.save(f, tick_data['points_dqn'])
        Image.fromarray(tick_data['topdown']).save(sp / 'topdown' / f'{frame}.png')


    # remove rgb cameras from base agent
    def sensors(self):
        sensors = super().sensors() 
        
        # get rid of old rgb cameras
        rgb, _, _, imu, gps, speed, topdown = sensors
        sensors = [rgb, imu, gps, speed, topdown]
        return sensors

    def tick(self, input_data):
        result = super().tick(input_data)

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
        #nodes = np.clip(nodes, 0, 256)
        commands = [command for _, command in route]
        target = np.clip(nodes[1], 0, 256)

        # populate results
        result['num_waypoints'] = len(route)
        result['commands'] = commands
        result['target'] = target
        result['route_map'] = nodes

        return result

    @torch.no_grad()
    def run_step(self, input_data, timestamp):
        if not self.initialized:
            self._init()

        tick_data = self.tick(input_data)

        # prepare inputs
        topdown = Image.fromarray(tick_data['topdown'])
        topdown = topdown.crop((128, 0, 128+256, 256))
        topdown = np.array(topdown)
        topdown = preprocess_semantic(topdown)
        topdown = topdown[None].cuda()
        tick_data['topdown_pth'] = topdown.clone().cpu()
        target = torch.from_numpy(tick_data['target'])
        target = target[None].cuda()

        # expectation points
        points_exp, vmap, hmap = self.net.forward(topdown, target, debug=True) # world frame
        points_exp = points_exp.clone().cpu().squeeze().numpy()
        points_exp = (points_exp + 1) / 2 * 256 # (-1,1) to (0,256)
        points_exp = np.clip(points_exp, 0, 256)

        # dqn points
        points_dqn, Q_all = self.net.get_dqn_actions(vmap, explore=self.burn_in) # (1,4,2), (1,4,1)
        points_dqn = np.clip(points_dqn.clone().cpu().squeeze().numpy(), 0, 256) # (4,2)
        tick_data['points_dqn'] = points_dqn
        tick_data['points_exp'] = points_exp
        tick_data['maps'] = (vmap, hmap)

        if self.aconfig.mode == 'dagger': # dagger and dqn
            points_lbc, _, _ = self.expert.forward(topdown, target, debug=True)
            points_lbc = points_lbc.clone().cpu().squeeze().numpy()
            points_lbc = (points_lbc + 1) / 2 * 256
            points_lbc = np.clip(points_lbc, 0, 256)
            tick_data['points_lbc'] = points_lbc

        if self.aconfig.mode == 'forward' or self.burn_in:
            #points_map = np.random.randint(0, 256, size=(4,2)) 
            x = np.random.randint(128 - 56, 128 + 56, (4,1))
            y = np.random.randint(256 - 128, 256, (4,1))
            points_map = np.hstack((x,y))

        if self.aconfig.mode == 'data': # expert behavior policy
            tick_data['points_lbc'] = points_exp
            points_map = points_exp
        else:
            points_map = points_dqn
        tick_data['points_map'] = points_map

        # get aim and controls
        points_world = self.converter.map_to_world(torch.Tensor(points_map)).numpy()
        aim = (points_world[1] + points_world[0]) / 2.0
        desired_speed = np.linalg.norm(points_world[0] - points_world[1]) * 2.0

        angle = np.degrees(np.pi / 2 - np.arctan2(aim[1], aim[0])) / 90
        steer = self._turn_controller.step(angle)
        steer = np.clip(steer, -1.0, 1.0)
        speed = tick_data['speed']
        delta = np.clip(desired_speed - speed, 0.0, 0.25)
        brake = desired_speed < 0.4 or (speed / desired_speed) > 1.1
        throttle = self._speed_controller.step(delta)
        throttle = np.clip(throttle, 0.0, 0.75)
        throttle = throttle if not brake else 0.0

        control = carla.VehicleControl()
        control.steer = steer
        control.throttle = throttle
        control.brake = float(brake)
        #print(timestamp) # GAMETIME

        condition = not self.burn_in and not self.aconfig.mode == 'forward' # not random
        condition = condition and (HAS_DISPLAY or self.config.save_debug)
        if condition:
            self.debug_display(tick_data, steer, throttle, brake, desired_speed)

        if self.config.save_data:
            self.save_data(tick_data)
        self.tick_data = tick_data
        return control

    def debug_display(self, tick_data, steer, throttle, brake, desired_speed, r=2):

        # 
        text_color = (255,255,255)
        points_color = (32,178,170) # purple
        aim_color = (127,255,212) # cyan
        route_colors = [(0,255,0), (255,255,255), (255,0,0), (255,0,0)] 
        
        vmap, hmap = tick_data['maps'] # (1, H, W)
        hmap = hmap.clone().cpu().numpy().squeeze() # (H,W)
        target_hmap = hmap / np.amax(hmap) * 256 # to grayscale
        target_hmap = target_hmap.astype(np.uint8)

        # rgb image on left, topdown w/vmap image on right
        # plot Q points instead of LBC version (argmax instead of expectation)
        points_map = tick_data['points_map']
        points_cam = self.converter.map_to_cam(torch.Tensor(points_map)).numpy()
        points_exp = tick_data['points_exp']

        route_map = tick_data['route_map']
        route_cam = self.converter.map_to_cam(torch.Tensor(route_map)).numpy()

        # (H,W,3) right image
        fused = fuse_vmaps(tick_data['topdown_pth'], vmap, temperature=10, alpha=1.0).squeeze()
        fused = Image.fromarray(fused)
        draw = ImageDraw.Draw(fused)
        for x, y in points_map[0:2]: # agent points
            draw.ellipse((x-r, y-r, x+r, y+r), points_color)
        x, y = np.mean(points_map[0:2], axis=0)
        draw.ellipse((x-r, y-r, x+r, y+r), aim_color)
        for x, y in points_exp:
            draw.ellipse((x-r, y-r, x+r, y+r), (0,255,127))
        for i, (x,y) in enumerate(route_map[:3]):
            #if x < 5 or x > 250 or y < 5 or y > 140: continue
            #if y > 140: continue
            draw.ellipse((x-2*r, y-2*r, x+2*r, y+2*r), route_colors[i])
        fused = np.array(fused)

        # left image
        rgb = Image.fromarray(tick_data['rgb'])
        draw = ImageDraw.Draw(rgb)
        for x,y in points_cam[0:2]:
            draw.ellipse((x-r, y-r, x+r, y+r), points_color)
        x, y = np.mean(points_cam[0:1], axis=0)
        draw.ellipse((x-r, y-r, x+r, y+r), aim_color)
        points_exp = self.converter.map_to_cam(torch.Tensor(points_exp)).numpy()
        for x, y in points_exp:
            draw.ellipse((x-r, y-r, x+r, y+r), (0,255,127))
        for i, (x,y) in enumerate(route_cam[:4]):
            if i == 0: # maybe skip first route map waypoint 
                xt, yt = route_map[0]
                if xt < 5 or xt > 250 or yt < 5 or yt > 250: continue
                #yt > 250: continue
            if x < 5 or x > 250 or y < 5 or y > 140: continue
            draw.ellipse((x-r, y-r, x+r, y+r), route_colors[i])
        rgb = Image.fromarray(np.array(rgb.resize((456, 256))))
        draw = ImageDraw.Draw(rgb)

        # draw debug text
        text_color = (255,255,255) #darkmagenta
        draw.text((5, 10), 'Steer: %.3f' % steer, text_color)
        draw.text((5, 30), 'Throttle: %.3f' % throttle, text_color)
        draw.text((5, 50), 'Brake: %s' % brake, text_color)
        draw.text((5, 70), 'Speed: %.3f' % tick_data['speed'], text_color)
        draw.text((5, 90), 'Desired: %.3f' % desired_speed, text_color)
        cur_command, next_command = tick_data['commands'][:2]
        draw.text((5, 110), f'Current: {cur_command}', text_color)
        draw.text((5, 130), f'Next: {next_command}', text_color)
        rgb = np.array(rgb)

        _combined = cv2.cvtColor(np.hstack((rgb, fused)), cv2.COLOR_BGR2RGB)
        self.debug_img = _combined

        if self.config.save_debug and self.step % 5 == 0:

            frame_number = self.step // 5
            save_path = self.save_path / 'debug' / f'{frame_number:06d}.png'
            cv2.imwrite(str(save_path), _combined)

        if HAS_DISPLAY:
            cv2.imshow('debug', _combined)
            cv2.waitKey(1)
        

    def destroy(self):
        self.sensor_interface = SensorInterface()

