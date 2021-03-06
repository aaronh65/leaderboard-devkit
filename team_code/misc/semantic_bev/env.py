import signal
import time, os
import gym
import numpy as np
import itertools
from collections import deque
from itertools import islice

from team_code.rl.common.env_utils import *
from team_code.rl.common.base_env import BaseEnv
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider


class CarlaEnv(BaseEnv):

    def __init__(self, config, client, agent):
        super().__init__(config, client, agent)

        # RL params
        self.history_size = config.agent.history_size
        self.obs_dim = (config.agent.bev_size, config.agent.bev_size, self.history_size,)
        self.observation_space = gym.spaces.Box(
                low=0, high=255, shape=self.obs_dim,
                dtype=np.uint8)
        self.action_dim = (2,)
        self.action_space = gym.spaces.Box(
                low=-1, high=1, shape=self.action_dim, 
                dtype=np.float32)

        # set up blocking checks
        self.last_hero_transforms = deque()
        self.max_positions_len = 60 
        self.blocking_distance = 2.0
        self.target_idx = 0
        self.last_waypoint = 0

    def reset(self, log=None):
        super().reset(log)
        
        self.last_waypoint = 0
        self.last_hero_transforms = deque()
        self._get_hero_route()

        if log is not None:
            self.env_log['total_waypoints'] = len(self.route_waypoints)
            self.env_log['last_waypoint'] = 0

        return np.zeros(self.obs_dim)

    def _get_hero_route(self):

        # retrieve new hero route
        self.route = CarlaDataProvider.get_ego_vehicle_route()

        route_waypoints = [route_elem[0] for route_elem in self.route]
        self.route_waypoints = [self.map.get_waypoint(loc) for loc in route_waypoints]
        self.route_transforms = [waypoint_to_vector(wp) for wp in self.route_waypoints]
        self.route_transforms = np.array(self.route_transforms)
        self.forward_vectors = [wp.transform.get_forward_vector() for wp in self.route_waypoints]
        self.forward_vectors = np.array( [[v.x, v.y, v.z] for v in self.forward_vectors])

        self.route_len = len(self.route_transforms)


    def step(self, action):
        # ticks the scenario and makes visual with new semantic bev image and cached info
        super().step(action) 
        if self.frame > 1:
            self.hero_agent.make_visualization()

        
        # check blocked and timeout
        info = {}
        self.aux_info = {}
        self.aux_info['timeout_done'] = self.frame >= 6000 # remove?
        hero_transform = CarlaDataProvider.get_transform(self.hero_actor)
        self.aux_info['blocked_done']= self._check_blocked(hero_transform)

        # get target and compute reward
        target_idx = self._get_target(hero_transform) # idxs
        target_waypoint = self.route_waypoints[target_idx]
        reward_info = self._get_reward(hero_transform, target_waypoint)

        criteria = [
                self.aux_info['blocked_done'], 
                self.aux_info['timeout_done'], 
                self.aux_info['distance_done']]
        done = any(criteria)

        
        info = {'is_success': reward_info['success']}
        info.update(self.aux_info)

        # set up experience from last step
        maps = self.hero_agent.cached_maps
        obs = np.stack(islice(maps, 1, self.history_size + 1), axis=2)
        action = self.hero_agent.cached_prev_action
        reward = self.hero_agent.cached_rinfo['reward']
        new_obs = np.stack(islice(maps, 0, self.history_size), axis=2)
        done = done or self.hero_agent.cached_done
        self.exp = [obs, action, reward, new_obs, done, info]

        # update hero cache for next step
        self.hero_agent.cached_rinfo = reward_info
        self.hero_agent.cached_done = done

        return new_obs, reward, done, info

    def _check_blocked(self, hero_transform):

        if self.frame > 60:
            if len(self.last_hero_transforms) < self.max_positions_len:
                #self.last_hero_transforms.append(hero_vector[:2])
                self.last_hero_transforms.append(hero_transform)
            else:
                self.last_hero_transforms.popleft()
                self.last_hero_transforms.append(hero_transform)
                start = self.last_hero_transforms[0].location
                end = self.last_hero_transforms[-1].location
                traveled = ((end.x-start.x)**2 + (end.y-start.y)**2)**0.5
                if traveled < self.blocking_distance:
                    return True
                    
        return False

    def _get_target(self, hero_transform, visualize=False):

        winsize = 100
        end_idx = min(self.last_waypoint + winsize, self.route_len)
        route_transforms = self.route_transforms[self.last_waypoint:end_idx]

        # distance info
        hero_transform_vec = transform_to_vector(hero_transform)
        hero2pt = route_transforms[:,:3] - hero_transform_vec[:3] # Nx3
        hero_fvec = hero_transform.get_forward_vector()
        hero_fvec = np.array([cvector_to_array(hero_fvec)]).T # 3x1

        # reachable criteria - does it take a >90 deg turn to get to waypoint?
        R_world2hero = np.array(hero_transform.get_inverse_matrix())[:3,:3]
        heading_vector = np.matmul(R_world2hero, hero2pt.T).T # Nx3
        y, x = heading_vector[:,1], heading_vector[:,0] # in carla coordinate system
        heading_angles = np.arctan2(y, x) * 180 / np.pi
        reachable = np.abs(heading_angles) < 90

        # aligned criteria - is the waypoint pointing the same direction we are?
        yaw_diffs = route_transforms[:, 4] - hero_transform_vec[4]
        yaw_diffs = (yaw_diffs + 180) % 360 - 180
        yaw_diffs = np.array([np.abs(yaw_diffs)]).flatten()
        aligned = yaw_diffs < 120 # some sharp right/left turns are > 90 degrees

        # replace with an np.all call?
        criteria = np.array([reachable, aligned]) # 2xN
        valid = np.prod(criteria, axis=0).flatten().astype(bool)
        valid_indices = np.arange(len(route_transforms))[valid]

        if len(valid_indices) == 0:
            self.aux_info['new_target'] = True
            self.aux_info['distance_done'] = False

            return self.last_waypoint

        # retrieve target
        target_idx = self.last_waypoint + valid_indices[0]
        self.aux_info['new_target'] = target_idx != self.last_waypoint
        target = self.route_waypoints[target_idx]
        self.last_waypoint = target_idx

        # check for distance
        tgt2hero = -np.array([hero2pt[valid_indices[0]]]).T # 3x1
        R_world2tgt = np.array(target.transform.get_inverse_matrix())[:3,:3]
        tgt2hero = np.matmul(R_world2tgt, tgt2hero).flatten()

        long_dist, lat_dist = np.abs(tgt2hero[:2])
        done = lat_dist > 4 or long_dist > 8 # 3/2 lane widths away from the center
        self.aux_info['distance_done'] = done

        # visualize
        draw_hero_route(self.world, self.route_waypoints, self.route_len, target_idx)
        #start_draw = max(0, target_idx-25)
        #end_draw = min(self.route_len, target_idx+25)
        #draw_waypoints(
        #        self.world, self.route_waypoints[start_draw:end_draw], 
        #        color=(0,0,255), life_time=0.06)
        
        self.env_log['last_waypoint'] = int(self.last_waypoint)

        if visualize:
            draw_waypoints(self.world, [target], color=(0,255,0), size=0.5)
            draw_arrow(self.world, hero_transform.location,
                    target.transform.location, color=(255,0,0), size=0.5)

        return target_idx
    
    def _get_reward(self, hero_transform, target_waypoint):
        hero = transform_to_vector(hero_transform)
        target = waypoint_to_vector(target_waypoint)

        # distance reward
        #dist_max = (4**2 + self.config.hop_resolution**2)**0.5
        #dist = min(np.linalg.norm(hero[:3] - target[:3]), dist_max)
        #dist_reward = (dist/dist_max - 1)**2 - 1
        #dist_reward = 0 - min(dist/dist_max, 1)
        lat_max = 4
        tgt2hero = hero[:3] - target[:3]
        R_world2tgt = np.array(target_waypoint.transform.get_inverse_matrix())[:3,:3]
        tgt2hero = np.matmul(R_world2tgt, tgt2hero).flatten()
        long_dist, lat_dist = np.abs(tgt2hero[:2])
        dist_reward = 0 - min(lat_dist/lat_max, 1)


        # rotation reward
        yaw_diff = (hero[4]-target[4]) % 360
        yaw_diff = yaw_diff if yaw_diff < 180 else 360 - yaw_diff
        yaw_max = 90
        yaw_frac = min(yaw_diff/yaw_max, 1)
        #yaw_reward = -yaw_frac**2 + 1
        yaw_reward = 1 - min(yaw_diff/yaw_max, 1)

        # speed reward
        hvel = CarlaDataProvider.get_velocity(self.hero_actor) # m/s
        hvel = hvel * 3600 / 1000 # km/h
        tvel = 40 # km/h
        vel_diff = abs(hvel-tvel)
        vel_reward = 1 - min(vel_diff/tvel, 1)


        # route reward
        route_reward = self.aux_info['new_target'] * 5 # 5 if new target else 0
        if self.aux_info['blocked_done'] or self.aux_info['distance_done']:
            route_reward = 10 if self.last_waypoint == self.route_len-1 else -5

        reward = dist_reward + yaw_reward + vel_reward + route_reward
        #reward = dist_reward + vel_reward + route_reward
        reward_info = {
                'reward': reward, 
                'dist_reward': dist_reward,
                'vel_reward': vel_reward,
                'yaw_reward': yaw_reward,
                'route_reward': route_reward,
                'success': self.last_waypoint > self.route_len-10,
                }

        if np.isnan(reward):
            reward_info['reward'] = 0
            print('caught NaN, setting reward to 0')
            print(reward_info)
        return reward_info
