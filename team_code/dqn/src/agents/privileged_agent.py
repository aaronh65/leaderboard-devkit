import os, yaml, copy
import cv2
import numpy as np
import torch

from PIL import Image, ImageDraw
from pathlib import Path
from carla import VehicleControl

from misc.utils import *
from lbc.carla_project.src.map_model import MapModel as LBCModel
from lbc.carla_project.src.dataset import preprocess_semantic
from lbc.carla_project.src.converter import Converter
from lbc.carla_project.src.common import CONVERTER, COLOR
from lbc.src.pid_controller import PIDController
from lbc.src.map_agent import MapAgent
from dqn.src.agents.map_model import fuse_logits
from leaderboard.envs.sensor_interface import SensorInterface

HAS_DISPLAY = int(os.environ.get('HAS_DISPLAY', 0))
text_color = (255,255,255)
aim_color = (60,179,113) # dark green
student_color = (65,105,225) # dark blue
expert_color = (178,34,34) # dark red
route_colors = [(255,255,255), (112,128,144), (47,79,79), (47,79,79)] 


def get_entry_point():
    return 'PrivilegedAgent'

class PrivilegedAgent(MapAgent):
    def setup(self, path_to_conf_file):
        super().setup(path_to_conf_file)

        self.econfig = dict_to_sns(self.config.env)
        self.aconfig = dict_to_sns(self.config.agent)
        #self.save_root = Path(self.config.save_root)

        with open(f'{self.config.save_root}/train_config.yml', 'r') as f:
            self.train_config = dict_to_sns(yaml.load(f, Loader=yaml.Loader))
        #self.control_type = self.train_config.control_type
        if self.train_config.control_type == 'learned':
            from dqn.src.agents.lc_map_model import MapModel
        elif self.train_config.control_type == 'points':
            from dqn.src.agents.map_model import MapModel

        self.converter = Converter()

        weights_path = self.aconfig.weights_path
        self.net = MapModel.load_from_checkpoint(weights_path)
        self.net.cuda()
        self.net.eval()

        if self.aconfig.dagger_expert:
            expert_path = Path(self.aconfig.expert_path)
            self.expert = LBCModel.load_from_checkpoint(expert_path)
            self.expert.cuda()
            self.expert.eval()

        if self.aconfig.safety_driver:
            from team_code.lbc.src.auto_pilot import AutoPilot
            self.safety_driver = AutoPilot(f'{self.config.project_root}/team_code/lbc/config/offline_autopilot.yml')


        self.burn_in = False

    def _init(self):
        super()._init()

        self._turn_controller = PIDController(K_P=1.25, K_I=0.75, K_D=0.3, n=40)
        self._speed_controller = PIDController(K_P=5.0, K_I=0.5, K_D=1.0, n=40)

        if self.config.save_debug or self.config.save_data:

            route_name = os.environ['ROUTE_NAME']
            repetition = os.environ['REPETITION']
            self.save_path = Path(f'{self.config.save_root}/data/{route_name}/{repetition}')
            self.save_path.mkdir(parents=True, exist_ok=False)
            print(self.save_path)

        if self.config.save_debug:
            (self.save_path / 'debug').mkdir()

        if self.config.save_data:
            (self.save_path / 'topdown').mkdir()
            (self.save_path / 'points_expert').mkdir()
            (self.save_path / 'points_student').mkdir()
            (self.save_path / 'measurements').mkdir()

        if self.aconfig.safety_driver:
            self.safety_driver._global_plan_world_coord = self._global_plan_world_coord
            self.safety_driver._global_plan= self._global_plan
            self.safety_driver._plan_HACK = self._plan_HACK
            self.safety_driver._plan_gps_HACK = self._plan_gps_HACK


    def reset(self):
        self.initialized = False
        
    def save_data(self,  tick_data):
        sp = self.save_path
        frame = f'{self.step:06d}'
        if 'points_expert' in tick_data.keys():
            with open(sp / 'points_expert' / f'{frame}.npy', 'wb') as f:
                np.save(f, tick_data['points_expert'])
        with open(sp / 'points_student' / f'{frame}.npy', 'wb') as f:
            np.save(f, tick_data['points_map'])
        Image.fromarray(tick_data['topdown']).save(sp / 'topdown' / f'{frame}.png')

    # remove rgb cameras from base agent
    def sensors(self):
        sensors = super().sensors() 
        
        # get rid of rgb cameras
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

        if self.aconfig.safety_driver:
            input_copy = copy.deepcopy(input_data)
            time_copy = copy.deepcopy(timestamp)
            safe_control = self.safety_driver.run_step(input_copy, time_copy, noviz=True)

        tick_data = self.tick(input_data)

        # prepare inputs
        topdown = Image.fromarray(tick_data['topdown'])
        topdown = topdown.crop((128, 0, 128+256, 256))
        topdown = np.array(topdown)
        topdown = preprocess_semantic(topdown)
        topdown = topdown[None].cuda()
        tick_data['topdown_processed'] = topdown.clone().cpu()

        target = torch.from_numpy(tick_data['target'])
        target = target[None].cuda()

        points, Qmap, tmap = self.net.forward(topdown, target)

        # 1. is the model using argmax or soft argmax?
        #if self.aconfig.waypoint_mode == 'softargmax':
        #    points_map = points.clone().cpu().squeeze().numpy()
        #    points_map = (points_map + 1) / 2 * 256 # (-1,1) to (0,256)
        #elif self.aconfig.waypoint_mode == 'argmax':
        #points_map, _ = self.net.get_argmax_actions(Qmap) # (1,4,2),(1,4,1)
        #points_map = points_map.clone().cpu().squeeze().numpy()
        #points_map = np.clip(points_map, 0, 256)
        #points_world = self.converter.map_to_world(torch.Tensor(points_map)).numpy()
        #aim = (points_world[1] + points_world[0]) / 2.0

        maps = [tmap, Qmap]
        if self.train_config.control_type == 'learned':
            
            control_out, Qmap_lc = self.net.get_control_from_points(points)
            control_out = control_out.detach().cpu().numpy().flatten().astype(float)
            maps.append(Qmap_lc)

            steer = control_out[0]
            if self.train_config.throttle_mode == 'throttle':
                brake = False
                throttle = control_out[1]
                desired_speed = 0
            elif self.train_config.throttle_mode == 'speed':
                desired_speed = control_out[1] * 10
                speed = tick_data['speed']
                brake = desired_speed < 0.4 or (speed / desired_speed) > 1.1
                delta = np.clip(desired_speed - speed, 0.0, 0.25)
                throttle = self._speed_controller.step(delta)
                throttle = np.clip(throttle, 0.0, 0.75)
                throttle = throttle if not brake else 0.0

            # for plotting purposes
            points_map = (points+1)/2*256
            points_map = points_map.clone().cpu().squeeze().numpy()
            points_map = np.clip(points_map, 0, 256)
            points_world = self.converter.map_to_world(torch.Tensor(points_map)).numpy()
            aim = (points_world[1] + points_world[0]) / 2.0

        elif self.train_config.control_type == 'points':
            points_map, _ = self.net.get_argmax_actions(Qmap) # (1,4,2),(1,4,1)
            points_map = points_map.clone().cpu().squeeze().numpy()
            points_map = np.clip(points_map, 0, 256)
            points_world = self.converter.map_to_world(torch.Tensor(points_map)).numpy()
            aim = (points_world[1] + points_world[0]) / 2.0

            angle = np.degrees(np.pi / 2 - np.arctan2(aim[1], aim[0])) / 90
            steer = self._turn_controller.step(angle)
            steer = np.clip(steer, -1.0, 1.0)

            desired_speed = np.linalg.norm(points_world[0] - points_world[1]) * 2.0
            speed = tick_data['speed']
            brake = desired_speed < 0.4 or (speed / desired_speed) > 1.1
            delta = np.clip(desired_speed - speed, 0.0, 0.25)
            throttle = self._speed_controller.step(delta)
            throttle = np.clip(throttle, 0.0, 0.75)
            throttle = throttle if not brake else 0.0

        control = VehicleControl()
        control.steer = steer
        control.throttle = throttle
        #control.brake = float(brake)
        control.brake = safe_control.brake if self.aconfig.safety_driver else float(brake)

        # debug mode - agent moves forward
        if self.aconfig.forward == True:
            #control.steer = np.random.random()*2-1
            control.steer = 0
            control.throttle = 1
            control.brake = 0

        # is there an expert present?
        if self.aconfig.dagger_expert is True : # dagger and dqn
            points_expert, _, _ = self.expert.forward(topdown, target)
            points_expert = points_expert.clone().cpu().squeeze().numpy()
            points_expert = (points_expert + 1) / 2 * 256
            points_expert = np.clip(points_expert, 0, 256)
            tick_data['points_expert'] = points_expert

        tick_data['maps'] = maps
        tick_data['points_map'] = points_map
        tick_data['aim_world'] = aim


        condition = not self.burn_in and not self.aconfig.forward == True # not random
        condition = condition and (self.config.save_debug and self.step % 4 == 0)
        if condition or HAS_DISPLAY:
            self.debug_display(tick_data, steer, throttle, brake, desired_speed)

        if self.config.save_data:
            self.save_data(tick_data)
        self.tick_data = tick_data
        return control

    def debug_display(self, tick_data, steer, throttle, brake, desired_speed, r=2):

        # rgb image on left, topdown w/vmap image on right
        # plot Q points instead of LBC version (argmax instead of expectation)
        aim = np.array(tick_data['aim_world'])
        route_map = tick_data['route_map']
        points_map = tick_data['points_map']
        points_expert = None if 'points_expert' not in tick_data.keys() else tick_data['points_expert']
        
        merge_list = []
        maps = tick_data['maps']
        if len(maps) == 2:
            tmap, Qmap = tick_data['maps'] # (1, H, W)
        else:
            tmap, Qmap, Qmap_lc = tick_data['maps'] # (1, H, W)
            Qmap_lc = spatial_norm(Qmap_lc)
            Qmap_lc = Qmap_lc[0][0].detach().cpu().numpy()
            Qmap_im = np.expand_dims(Qmap_lc, -1)
            Qmap_im = np.tile(Qmap_im, (1,1,3))
            Qmap_im = Image.fromarray(np.uint8(Qmap_im*255))

            Qmap_draw = ImageDraw.Draw(Qmap_im)
            dspeed = desired_speed / 10
            x = (steer + 1) / 2 * (self.train_config.n_steer-1)
            y = (1-dspeed) * (self.train_config.n_throttle-1)
            Qmap_draw.ellipse((x,y,x+1,y+1), (0,0,255))
            Qmap_im = Qmap_im.resize((256,256), resample=0)
            merge_list.append(Qmap_im)

        #cv2.imshow('Qmap_lc', Qmap_lc)
        #cv2.waitKey(0)

        # (H,W,3) middle image if it exists
        topdown = tick_data['topdown_processed']
        fused = fuse_logits(topdown, Qmap).squeeze()
        fused = Image.fromarray(fused)
        draw = ImageDraw.Draw(fused)
        for x, y in points_map:
            draw.ellipse((x-r, y-r, x+r, y+r), student_color)
        if points_expert is not None:
            for x, y in points_expert:
                draw.ellipse((x-r, y-r, x+r, y+r), expert_color)
        x, y = self.converter.world_to_map(torch.Tensor(aim)).numpy()

        draw.ellipse((x-r, y-r, x+r, y+r), aim_color)
        for i, (x,y) in enumerate(route_map[:3]):
            draw.ellipse((x-2*r, y-2*r, x+2*r, y+2*r), route_colors[i])
        fused = np.array(fused)
        merge_list.append(fused)

        # middle image
        aim_cam = self.converter.world_to_cam(torch.Tensor(aim)).numpy()
        route_cam = self.converter.map_to_cam(torch.Tensor(route_map)).numpy()
        points_cam = self.converter.map_to_cam(torch.Tensor(points_map)).numpy()

        rgb = Image.fromarray(tick_data['rgb'])
        draw = ImageDraw.Draw(rgb)
        for x,y in points_cam:
            draw.ellipse((x-r, y-r, x+r, y+r), student_color)
        if points_expert is not None:
            points_expert_cam = self.converter.map_to_cam(torch.Tensor(points_expert)).numpy()
            for x, y in points_expert_cam:
                draw.ellipse((x-r, y-r, x+r, y+r), expert_color)
        x, y = aim_cam
        draw.ellipse((x-r, y-r, x+r, y+r), aim_color)
        for i, (x,y) in enumerate(route_cam[:4]):
            if i == 0: # maybe skip first route map waypoint 
                xt, yt = route_map[0]
                if xt < 5 or xt > 250 or yt < 5 or yt > 250: continue
            if x < 5 or x > 250 or y < 5 or y > 140: continue
            draw.ellipse((x-2*r, y-2*r, x+2*r, y+2*r), route_colors[i])
        rgb = Image.fromarray(np.array(rgb.resize((456, 256)))) # match height to map
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
        merge_list.append(rgb)

        _combined = cv2.cvtColor(np.hstack(merge_list), cv2.COLOR_RGB2BGR)
        self.debug_img = _combined

        if self.config.save_debug and self.step % 4 == 0:

            frame_number = self.step // 4
            save_path = self.save_path / 'debug' / f'{frame_number:06d}.png'
            cv2.imwrite(str(save_path), _combined)

        if HAS_DISPLAY:
            cv2.imshow('debug', _combined)
            cv2.waitKey(1)

    def destroy(self):
        self.sensor_interface = SensorInterface()

