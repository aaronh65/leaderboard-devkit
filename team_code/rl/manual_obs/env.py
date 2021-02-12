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
        self.nstate_waypoints = config.agent.num_state_waypoints
        self.waypoint_state_dim = config.agent.waypoint_state_dim
        self.obs_dim = (self.waypoint_state_dim + 4,) 
        #self.obs_dim = (self.waypoint_state_dim + 5,)
        self.observation_space = gym.spaces.Box(
                low=-1, high=1, shape=self.obs_dim, 
                dtype=np.float32)
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
        max_dist = (4**2 + (self.config.hop_resolution*(self.nstate_waypoints+1))**2)**0.5
        self.obs_norm = np.array([max_dist, 180, 5, 180]) # distance, heading, z, dyaw

    
    def reset(self, log=None):
        super().reset(log)
        
        self.last_waypoint = 0
        self.last_traffic_light = None
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
        # ticks the scenario
        super().step(action) 

        # check blocked and timeout
        info = {}
        hero_transform = CarlaDataProvider.get_transform(self.hero_actor)
        blocked_done = self._check_blocked(hero_transform)
        timeout_done = self.frame >= 6000 # remove?

        # get target and next observation
        target_idx, distance_done = self._get_target(hero_transform) # idxs
        target_waypoint = self.route_waypoints[target_idx]
        obs = self._get_observation(hero_transform, target_idx)
        
        # compute reward and visualize
        reward_info = self._get_reward(hero_transform, target_waypoint, distance_done or blocked_done)
        self.hero_agent.cached_rinfo = reward_info
        self.hero_agent.make_visualization(self.obs_norm)

        draw_waypoints(self.world, [target_waypoint], color=(0,255,0), size=0.5)
        draw_arrow(self.world, hero_transform.location,
                target_waypoint.transform.location, color=(255,0,0), size=0.5)
        

        criteria = [blocked_done, timeout_done, distance_done]
        done = any(criteria)
        #info = {'blocked': blocked_done, 'timeout': timeout_done, 'too_far': distance_done}
        info = {'is_success': reward_info['success']}
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

    def _get_target(self, hero_transform):

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
        return target_idx, done
    
    
    def _get_observation(self, hero_transform, target_idx):

        obs = np.zeros(self.obs_dim)

        # state per waypoint (x,y,z in agent frame + yaw diff) and agent velocity
        wstates = np.zeros((self.nstate_waypoints, self.waypoint_state_dim))
        for i in range(self.nstate_waypoints):

            idx = min(len(self.route_waypoints)-1, target_idx + i)
            wpt = self.route_waypoints[idx]
            wstates[i] = self._get_waypoint_state(hero_transform, wpt)
            if i != 0:
                draw_waypoints(self.world, [wpt], color=(0,100,100), size=0.5)
        norm_wstate = np.mean(wstates, axis=0)
        norm_wstate = norm_wstate / self.obs_norm # heading and z are already [-1, 1] after clipping
        norm_wstate[0] = norm_wstate[0] * 2 - 1 # from [0, 1] to [-1, 1]
        norm_wstate = np.clip(norm_wstate, -1, 1)

        # average curvature
        max_curvature = 0
        for i in range(self.nstate_waypoints-1):
            idx = target_idx + i
            if idx >= len(self.route_waypoints) - 1:
                max_curvature += 0
            else:
                location1 = waypoint_to_vector(self.route_waypoints[target_idx])[:3]
                location2 = waypoint_to_vector(self.route_waypoints[idx+1])[:3]
                ref2tgt = np.array([location2-location1]).T # 3x1

                R_world2ref = np.array(self.route_waypoints[target_idx].transform.get_inverse_matrix())
                R_world2ref = R_world2ref[:3,:3]
                ref2tgt = np.matmul(R_world2ref, ref2tgt).flatten()
                heading = np.arctan2(ref2tgt[1], ref2tgt[0]) * 180 / np.pi
                if abs(heading) > 120:
                    continue
                if abs(heading) > abs(max_curvature):
                    max_curvature = heading

        norm_curvature = max_curvature / 180
        norm_curvature = np.clip(norm_curvature, -1, 1)


        # velocity
        velocity = CarlaDataProvider.get_velocity(self.hero_actor)
        velocity = velocity * 3600 / 1000 # km/h
        norm_velocity = np.clip(velocity, 0, 80)
        norm_velocity = norm_velocity / 40 - 1 # squash to -1, 1

        # steer
        norm_steer = self.hero_agent.cached_control.steer # already [-1, 1]

        # completion
        completion = self.last_waypoint / self.route_len # fraction in [0, 1]
        norm_completion = completion * 2 - 1

        # traffic lights
        maybe_traffic_light = CarlaDataProvider.get_next_traffic_light(self.hero_actor)
        tl_dist = 1
        if maybe_traffic_light is not None:
            traffic_light, distance_to_light = maybe_traffic_light
            #print(distance_to_light)
            self.last_traffic_light = traffic_light
            state = self.last_traffic_light.state
            cmap = {
                    carla.TrafficLightState.Red: (255,0,0),
                    carla.TrafficLightState.Yellow: (255,255,0),
                    carla.TrafficLightState.Green: (0,255,0),}
            draw_transforms(
                    self.world, 
                    [traffic_light.get_transform()],
                    color=cmap[state],
                    z=2.0, life_time = 0.06, size=0.6)

            if state != carla.TrafficLightState.Green:
                tl_dist = distance_to_light / 12.5 - 1 # [0, 25] to [-1, 1]
                tl_dist = np.clip(tl_dist, -1, 1)

        #obs = np.hstack([norm_wstate, [norm_curvature, norm_velocity, norm_steer, norm_completion, tl_dist]])
        obs = np.hstack([norm_wstate, [norm_curvature, norm_velocity, norm_steer, norm_completion]])
        return obs

    def _get_waypoint_state(self, hero_transform, target_waypoint):

        # transform target from world frame to hero frame
        target = waypoint_to_vector(target_waypoint)
        x,y,z = target[:3]
        target_location = np.array([[x,y,z,1]]).T # 4x1
        world_to_hero = hero_transform.get_inverse_matrix() # 4x4
        target_in_hero = np.matmul(world_to_hero, target_location).flatten()
        target_in_hero = target_in_hero[:3]
        x,y,z = target_in_hero
        heading = np.arctan2(y,x) * 180 / np.pi
        flat_distance = (x**2 + y**2)**0.5

        dyaw = sgn_angle_diff(hero_transform.rotation.yaw, target_waypoint.transform.rotation.yaw)

        state = np.array([flat_distance, heading, z, dyaw])
        return state

    def _get_reward(self, hero_transform, target_waypoint, blocked_or_distance_done):
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
        lat_dist = np.clip(lat_dist, 0, 4)
        dist_reward = 0 - min(lat_dist/lat_max, 1)

        # rotation reward
        dyaw = np.abs(sgn_angle_diff(hero[4], target[4]))
        yaw_max = 180
        #yaw_reward = -yaw_frac**2 + 1
        yaw_reward = 1 - min(dyaw/yaw_max, 1)

        # speed reward
        hvel = CarlaDataProvider.get_velocity(self.hero_actor) # m/s
        hvel = hvel * 3600 / 1000 # km/h
        tvel = 40 # km/h
        vel_diff = abs(hvel-tvel)
        vel_reward = 1 - min(vel_diff/tvel, 1)

        # route reward
        route_reward = self.last_waypoint / self.route_len
        if blocked_or_distance_done:
            #route_reward = 1000 if self.last_waypoint == self.route_len-1 else -100
            route_reward = 100 if self.last_waypoint == self.route_len-1 else -10

        # traffic light reward
        traffic_reward = 0
        hero_waypoint = self.map.get_waypoint(self.hero_actor.get_location())
        if self.last_traffic_light is not None and hero_waypoint.is_intersection:
            if self.last_traffic_light.state == carla.TrafficLightState.Red:
                traffic_reward = -5

        #reward = dist_reward + yaw_reward + vel_reward + route_reward + traffic_reward
        reward = dist_reward + vel_reward + yaw_reward + route_reward
        reward_info = {
                'reward': reward, 
                'dist_reward': dist_reward,
                'vel_reward': vel_reward,
                'yaw_reward': yaw_reward,
                'route_reward': route_reward,
                'traffic_reward': traffic_reward,
                'success': self.last_waypoint > self.route_len-10}
        if np.isnan(reward):
            reward_info['reward'] = 0
            print('caught NaN, setting reward to 0')
            print(reward_info)
        return reward_info
