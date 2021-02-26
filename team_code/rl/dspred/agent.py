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
from rl.dspred.map_model import MapModel
from lbc.carla_project.src.dataset import preprocess_semantic
from lbc.carla_project.src.converter import Converter
from lbc.carla_project.src.common import CONVERTER, COLOR
from lbc.common.map_agent import MapAgent
from lbc.common.pid_controller import PIDController

from leaderboard.envs.sensor_interface import SensorInterface
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

DEBUG = int(os.environ.get('HAS_DISPLAY', 0))

def get_entry_point():
    return 'DSPredAgent'

class DSPredAgent(MapAgent):
    def setup(self, path_to_conf_file):
        super().setup(path_to_conf_file)
        self.converter = Converter()
        project_root = self.config.project_root
        weights_path = self.config.agent.weights_path
        self.net = MapModel.load_from_checkpoint(f'{project_root}/{weights_path}')
        self.net.cuda()
        self.net.eval()

    def _init(self):
        super()._init()

        self._turn_controller = PIDController(K_P=1.25, K_I=0.75, K_D=0.3, n=40)
        self._speed_controller = PIDController(K_P=5.0, K_I=0.5, K_D=1.0, n=40)

    def reset(self):
        if not self.config.env.save_data:
            ROUTE_NAME = os.environ.get('ROUTE_NAME', 0)
            self.save_debug_path = Path(f'{self.config.save_root}/images/{ROUTE_NAME}') 
            if not os.path.exists(str(self.save_debug_path)):
                os.makedirs(str(self.save_debug_path))

                rep_number = len(os.listdir(self.save_debug_path))
                self.rep_name = f'repetition_{rep_number:02d}'
                paths = [self.save_debug_path / self.rep_name / 'debug', self.save_debug_path / self.rep_name / 'heatmaps']
                for path in paths:
                    if not os.path.exists(str(path)):
                        os.makedirs(str(path))
        self.initialized = False

    # remove rgb cameras from base agent
    def sensors(self):
        sensors = super().sensors()
        #sensors = sensors[3:]
        return sensors

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
            print('reinitializing agent...')
            self._init()

        
        tick_data = self.tick(input_data)
        topdown = Image.fromarray(tick_data['topdown'])
        topdown = topdown.crop((128, 0, 128+256, 256))
        topdown = np.array(topdown)
        map_view = COLOR[CONVERTER[topdown.copy()]]
        tick_data['map_view'] = map_view
        #cv2.imshow('topdown', topdown_view)

        topdown = preprocess_semantic(topdown)
        topdown = topdown[None].cuda()

        target = torch.from_numpy(tick_data['target'])
        target = target[None].cuda()

        points, hmaps = self.net.forward(topdown, target, debug=True) # world frame
        tick_data['hmaps'] = hmaps
        points_map = points.clone().cpu().squeeze()
        # what's this conversion for?
        # was this originally normalized for training stability or something?
        points_map = points_map + 1
        points_map = points_map / 2 * 256
        points_map = np.clip(points_map, 0, 256)
        points_cam = self.converter.map_to_cam(points_map).numpy()
        points_world = self.converter.map_to_world(points_map).numpy()
        points_map = points_map.numpy()

        tick_data['points_map'] = points_map
        tick_data['points_cam'] = points_cam
        tick_data['points_world'] = points_world

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

        if self.config.agent.forward:
            control = carla.VehicleControl()
            control.steer = 0
            control.throttle = 0.75
            control.brake = False

        self.aim = self.converter.world_to_map(torch.Tensor(aim)).numpy()
        self.obs = (tick_data['topdown'], tick_data['target'])

        if DEBUG or self.config.agent.save_images:

            # transform image model cam points to overhead BEV image (spectator frame?)
            self.debug_display(
                    tick_data, steer, throttle, brake, desired_speed)
            
        return control

    def debug_display(self, tick_data, steer, throttle, brake, desired_speed, r=2):

        colors = [(0,255,0), (255,0,0), (139,0,139), (139,0,139)] # for waypoints
        
        target_hmap = tick_data['hmaps'][1]
        target_hmap = target_hmap[0][0].clone().cpu().numpy()
        target_hmap = target_hmap / np.amax(target_hmap) * 256
        target_hmap = target_hmap.astype(np.uint8)

        zeros = np.zeros(target_hmap.shape)
        target_hmap = np.dstack((zeros, target_hmap, zeros))
        hmap_tgt = tick_data['map_view'].copy()
        hmap_tgt = hmap_tgt + target_hmap
        hmap_tgt = np.clip(hmap_tgt, a_min=0, a_max=255).astype(np.uint8)

        #cv2.imshow('target', target_hmap)


        points_map = tick_data['points_map']
        logit = tick_data['hmaps'][0]
        flat = logit.view(logit.shape[:-2] + (-1,))
        hmap = F.softmax(flat / self.net.net.temperature, dim=-1).view_as(logit)
        hmap = hmap[0].clone().cpu().numpy()
        
        alpha = 0.75
        hmap_stack = []
        for t in range(len(hmap)):
            hmap_out = hmap_tgt.copy()
            
            hmap_view = hmap[t] / np.amax(hmap[t]) * 256 # rescaling
            hmap_view = np.dstack((hmap_view, hmap_view, hmap_view))
            hmap_view = hmap_view.astype(np.uint8)
            cv2.addWeighted(hmap_view, alpha, hmap_out, 1, 0, hmap_out)
            #hmap_out = hmap_out.astype(np.uint8)
            hmap_im = Image.fromarray(hmap_out)
            hmap_imdraw = ImageDraw.Draw(hmap_im)
            x, y = points_map[t]
            hmap_imdraw.ellipse((x-r, y-r, x+r, y+r), (0, 191, 255))
            #hmap_view = np.expand_dims(hmap_view, 2) * alpha + topdown_view
            hmap_im = cv2.cvtColor(np.array(hmap_im), cv2.COLOR_BGR2RGB)
            hmap_stack.append(hmap_im)
            #cv2.imshow(f't={0.5*(t+1)}', hmap_im)
        hmap_comb = np.hstack(hmap_stack)

        topdown = tick_data['topdown']
        _topdown = Image.fromarray(COLOR[CONVERTER[topdown]])
        _topdown_draw = ImageDraw.Draw(_topdown)

        # model points
        points_td = points_map + [128, 0]
        for i, (x,y) in enumerate(points_td):
            _topdown_draw.ellipse((x-2*r, y-2*r, x+2*r, y+2*r), (0,191,255))
        route_map = tick_data['route_map']
        route_td = route_map + [128, 0]

        # control point
        aim_world = np.array(tick_data['aim_world'])
        aim_map = self.converter.world_to_map(torch.Tensor(aim_world)).numpy()
        aim_map = aim_map + [128,0]
        x, y = aim_map
        _topdown_draw.ellipse((x-2, y-2, x+2, y+2), (255, 105, 147))

        # route waypoints
        for i, (x, y) in enumerate(route_td[:3]):
            _topdown_draw.ellipse((x-2*r, y-2*r, x+2*r, y+2*r), colors[i])
        # make RGB images

        # draw center RGB image
        _rgb = Image.fromarray(tick_data['rgb'])
        _draw_rgb = ImageDraw.Draw(_rgb)
        for x, y in tick_data['points_cam']: # image model waypoints
            #x = (x + 1)/2 * 256
            #y = (y + 1)/2 * 144
            _draw_rgb.ellipse((x-2, y-2, x+2, y+2), (0, 191, 255))

        # transform aim from world to cam
        aim_cam = self.converter.world_to_cam(torch.Tensor(aim_world)).numpy()
        x, y = aim_cam
        _draw_rgb.ellipse((x-2, y-2, x+2, y+2), (255, 105, 147))

        # draw route waypoints in RGB image
        route_map = np.array(tick_data['route_map'])
        route_map = np.clip(route_map, 0, 256)
        route_map = route_map[:3].squeeze() # just the next couple
        route_cam = self.converter.map_to_cam(torch.Tensor(route_map)).numpy()
        for i, (x, y) in enumerate(route_cam):
            if i == 0: # waypoint we just passed
                if not (0 < y < 143 and 0 < x < 255):
                    continue
            _draw_rgb.ellipse((x-2, y-2, x+2, y+2), colors[i])

        _combined = Image.fromarray(np.hstack([tick_data['rgb_left'], _rgb, tick_data['rgb_right']]))
        _draw = ImageDraw.Draw(_combined)

        # draw debug text
        text_color = (255,255,255) #darkmagenta
        #text_color = (139, 0, 139) #darkmagenta
        _draw.text((5, 10), 'Steer: %.3f' % steer, text_color)
        _draw.text((5, 30), 'Throttle: %.3f' % throttle, text_color)
        _draw.text((5, 50), 'Brake: %s' % brake, text_color)
        _draw.text((5, 70), 'Speed: %.3f' % tick_data['speed'], text_color)
        _draw.text((5, 90), 'Desired: %.3f' % desired_speed, text_color)
        cur_command, next_command = tick_data['commands'][:2]
        _draw.text((5, 110), f'Current: {cur_command}', text_color)
        _draw.text((5, 130), f'Next: {next_command}', text_color)

        _rgb_img = _combined.resize((int(256/ _combined.size[1] * _combined.size[0]), 256))
        _topdown = _topdown.resize((256, 256))
        _save_img = Image.fromarray(np.hstack([_rgb_img, _topdown]))
        _save_img = cv2.cvtColor(np.array(_save_img), cv2.COLOR_BGR2RGB)

        if self.step % 10 == 0 and self.config.agent.save_images:
            frame_number = self.step // 10
            save_path = self.save_debug_path / self.rep_name / 'debug' / f'{frame_number:06d}.png'
            cv2.imwrite(str(save_path), _save_img)
            save_path = self.save_debug_path / self.rep_name / 'heatmaps' / f'{frame_number:06d}.png'
            cv2.imwrite(str(save_path), hmap_comb)
            #save_path = self.save_debug_path / self.rep_name / 'heatmaps' / f'{frame_number:06d}_tgt.png'
            #cv2.imwrite(str(save_path), cv2.cvtColor(hmap_tgt, cv2.COLOR_BGR2RGB))

        if DEBUG:
            cv2.imshow('heatmaps', hmap_comb)
            cv2.imshow('debug', _save_img)
            cv2.waitKey(1)
 
    def destroy(self):
        self.sensor_interface = SensorInterface()

