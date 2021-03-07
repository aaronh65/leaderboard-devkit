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
from rl.dspred.online_map_model import MapModel, fuse_vmaps
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

        self.burn_in = False

    def _init(self):
        super()._init()
        self._turn_controller = PIDController(K_P=1.25, K_I=0.75, K_D=0.3, n=40)
        self._speed_controller = PIDController(K_P=5.0, K_I=0.5, K_D=1.0, n=40)

    def reset(self):
        self.initialized = False

        # make debug directories
        save_root = self.config.save_root
        if self.config.save_debug:
            ROUTE_NAME, REPETITION = os.environ['ROUTE_NAME'], os.environ['REPETITION']
            self.debug_path = Path(f'{save_root}/debug/{ROUTE_NAME}/{REPETITION}')
            self.debug_path.mkdir(exist_ok=True, parents=True)

        # TODO: make data directories and reimplement data collection w/better file structure
        if self.config.save_data:
            pass

    # remove rgb cameras from base agent
    def sensors(self):
        sensors = super().sensors() 
        
        # get rid of old rgb cameras
        rgb, _, _, imu, gps, speed, topdown = sensors
        #rgb = {
        #    'type': 'sensor.camera.rgb',
        #    'x': 1.3, 'y': 0.0, 'z': 1.3,
        #    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
        #    'width': 456, 'height': 256, 'fov': 90,
        #    'id': 'rgb'
        #    }

        #rgb_topdown = \ 
        #    {'type': 'sensor.camera.rgb',
        #        'x': 0.0, 'y': 0.0, 'z': 100.0,
        #        'roll': 0.0, 'pitch': -90.0, 'yaw': 0.0,
        #        'width': 512, 'height': 512, 'fov': 5 * 10.0,
        #        'id': 'rgb_topdown'}
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
        topdown = Image.fromarray(tick_data['topdown'])
        topdown = topdown.crop((128, 0, 128+256, 256))
        topdown = np.array(topdown)
        topdown_save = topdown.copy()

        if not self.burn_in:

            # prepare inputs
            topdown = preprocess_semantic(topdown)
            topdown = topdown[None].cuda()
            tick_data['topdown_pth'] = topdown.clone().cpu()
            target = torch.from_numpy(tick_data['target'])
            target = target[None].cuda()

            # forward
            points, (vmap, hmap) = self.net.forward(topdown, target, debug=True) # world frame

            # retrieve action
            actions, Q_all = self.net.get_actions(vmap) # (1,4,2), (1,4)
            points_map = np.clip(actions.clone().cpu().squeeze(), 0, 256) # (4,2)

            # retrieve and transform points
            #points_map = points.clone().cpu().squeeze()
            #points_map = (points_map + 1) / 2 * 256
            points_map = np.clip(points_map, 0, 256)
            points_cam = self.converter.map_to_cam(points_map).numpy()
            points_world = self.converter.map_to_world(points_map).numpy()
            points_map = points_map.numpy()
                        
            tick_data['maps'] = (vmap, hmap)
            tick_data['points_cam'] = points_cam
            tick_data['points_map'] = points_map

        else: # burning in
            points_map = np.random.randint(0, 256, size=(4,2)) 
            points_world = self.converter.map_to_world(torch.Tensor(points_map)).numpy()
            #points_cam = self.converter.map_to_cam(torch.Tensor(points_map)).numpy()
            #print('burning in')
            #print((points_world[1] + points_world[0]) / 2.0)
            #print(np.linalg.norm(points_world[0] - points_world[1]) * 2.0)


        # get aim and controls
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

        if not self.burn_in and (HAS_DISPLAY or self.config.save_debug):
            self.debug_display(tick_data, steer, throttle, brake, desired_speed)

        
        self.action = points_map
        self.state = (tick_data['topdown'], tick_data['target'])

        return control

    def debug_display(self, tick_data, steer, throttle, brake, desired_speed, r=2):

        # 
        text_color = (255,255,255)
        #points_color = (138,43,226) # purple
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
        points_cam = tick_data['points_cam']
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
        for i, (x,y) in enumerate(route_map[:4]):
            draw.ellipse((x-r, y-r, x+r, y+r), route_colors[i])
        fused = np.array(fused)

        # left image
        rgb = Image.fromarray(tick_data['rgb'])
        draw = ImageDraw.Draw(rgb)
        for x,y in points_cam[0:2]:
            draw.ellipse((x-r, y-r, x+r, y+r), points_color)
        x, y = np.mean(points_cam[0:1], axis=0)
        draw.ellipse((x-r, y-r, x+r, y+r), aim_color)
        for i, (x,y) in enumerate(route_cam[:4]):
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

        if self.debug_path is not None:
            frame_number = self.step // 5
            save_path = self.debug_path / f'{frame_number:06d}.png'
            cv2.imwrite(str(save_path), _combined)

        if HAS_DISPLAY:
            cv2.imshow('debug', _combined)
            cv2.waitKey(1)

        return
        

    def destroy(self):
        self.sensor_interface = SensorInterface()

