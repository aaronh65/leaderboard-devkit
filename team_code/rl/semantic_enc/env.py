import signal
import time, os
import gym
import numpy as np
import itertools
from collections import deque

from team_code.rl.common.env_utils import *
from team_code.rl.common.base_env import BaseEnv
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider


class CarlaEnv(BaseEnv):

    def __init__(self, config, client, agent):
        super().__init__(config, client, agent)

        # RL params
        self.nstate_waypoints = config.sac.num_state_waypoints
        self.waypoint_state_dim = config.sac.waypoint_state_dim
        self.obs_dim = (256,256,3,)
        self.observation_space = gym.spaces.Box(
                low=-1, high=1, shape=self.obs_dim,
                dtype=np.uint8)
        self.action_dim = (2,)
        self.action_space = gym.spaces.Box(
                low=-1, high=1, shape=self.action_dim, 
                dtype=np.float32)

        # set up blocking checks
        self.last_hero_transforms = deque()
        self.max_positions_len = 60 
        self.blocking_distance = 3.0
        self.target_idx = 0
        self.last_waypoint = 0
        max_dist = (4**2 + (self.config.hop_resolution*(self.nstate_waypoints+1))**2)**0.5
        self.obs_norm = np.array([max_dist, 180, 5]) # distance, heading, z

        self.cached_experience = [None, None, None, None, None, {}] 

    
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

        # check blocked and timeout
        info = {}
        hero_transform = CarlaDataProvider.get_transform(self.hero_actor)
        blocked_done = self._check_blocked(hero_transform)
        #timeout_done = self.frame >= 6000 # remove?
        timeout_done = False

        # get target and compute reward
        target_idx, distance_done = self._get_target(hero_transform) # idxs
        target_waypoint = self.route_waypoints[target_idx]
        reward_info = self._get_reward(
                hero_transform, 
                target_waypoint, 
                distance_done or blocked_done)

        criteria = [blocked_done, timeout_done, distance_done]
        done = any(criteria)
        info = {'blocked': blocked_done, 'timeout': timeout_done, 'too_far': distance_done}

        # set up experience from last step
        if done:
            obs = self.hero_agent.cached_map
            action = self.hero_agent.cached_action
            reward = reward_info['reward']
            done = done
            new_obs = obs
        else:
            obs = self.hero_agent.cached_prev_map
            action = self.hero_agent.cached_prev_action
            reward = self.hero_agent.cached_rinfo['reward']
            done = self.hero_agent.cached_done
            new_obs = self.hero_agent.cached_map

        self.exp = [obs, action, reward, new_obs, done, {}]

        # update hero cache for next step
        self.hero_agent.cached_rinfo = reward_info
        self.hero_agent.cached_done = done

        return obs, reward_info['reward'], done, info

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
            return self.last_waypoint, True

        # retrieve target
        target_idx = self.last_waypoint + valid_indices[0]
        target = self.route_waypoints[target_idx]
        self.last_waypoint = target_idx

        # check for distance
        tgt2hero = -np.array([hero2pt[valid_indices[0]]]).T # 3x1
        R_world2tgt = np.array(target.transform.get_inverse_matrix())[:3,:3]
        tgt2hero = np.matmul(R_world2tgt, tgt2hero).flatten()

        long_dist, lat_dist = np.abs(tgt2hero[:2])
        done = lat_dist > 4 # 3/2 lane widths away from the center

        # visualize
        start_draw = max(0, target_idx-25)
        end_draw = min(self.route_len, target_idx+25)
        draw_waypoints(
                self.world, self.route_waypoints[start_draw:end_draw], 
                color=(0,0,255), life_time=0.06)
        
        self.env_log['last_waypoint'] = int(self.last_waypoint)

        if visualize:
            draw_waypoints(self.world, [target], color=(0,255,0), size=0.5)
            draw_arrow(self.world, hero_transform.location,
                    target.transform.location, color=(255,0,0), size=0.5)

        return target_idx, done
    
    def _get_reward(self, hero_transform, target_waypoint, blocked_or_distance_done):
        hero = transform_to_vector(hero_transform)
        target = waypoint_to_vector(target_waypoint)

        # distance reward
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
        route_reward = self.last_waypoint / self.route_len
        if blocked_or_distance_done:
            route_reward = 10 if self.last_waypoint == self.route_len-1 else -5

        #reward = dist_reward + yaw_reward + vel_reward + route_reward
        reward = dist_reward + vel_reward + route_reward
        reward_info = {
                'reward': reward, 
                'dist_reward': dist_reward,
                'vel_reward': vel_reward,
                'route_reward': route_reward}
        return reward_info
