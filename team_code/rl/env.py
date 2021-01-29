import signal
import time, os
import gym
import numpy as np
from collections import deque

from env_utils import *
from reward_utils import *
from carla import Client

from srunner.scenariomanager.timer import GameTime
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from leaderboard.scenarios.scenario_manager import ScenarioManager
from leaderboard.scenarios.route_scenario import RouteScenario
from leaderboard.utils.route_indexer import RouteIndexer

class CarlaEnv(gym.Env):

    def __init__(self, config, client, agent):
        super(CarlaEnv, self).__init__()

        self.config = config.env
        self.manager = ScenarioManager(60, False)
        self.scenario = None
        self.hero_actor = None

        # RL params
        self.nstate_waypoints = config.sac.num_state_waypoints
        self.waypoint_state_dim = config.sac.waypoint_state_dim
        self.obs_dim = self.nstate_waypoints*self.waypoint_state_dim + 2 # velocity + completion
        self.observation_space = gym.spaces.Box(
                #low=-1, high=1, shape=(6,), 
                low=-1, high=1, shape=(self.obs_dim,), 
                dtype=np.float32)
        self.action_dim = 3
        self.action_space = gym.spaces.Box(
                low=-1, high=1, shape=(self.action_dim,), 
                dtype=np.float32)

        self.agent_instance = agent
        
        # route indexer
        data_path = f'{config.project_root}/leaderboard/data'
        routes = f'{data_path}/{self.config.routes}'
        scenarios = f'{data_path}/{self.config.scenarios}'
        self.indexer = RouteIndexer(routes, scenarios, self.config.repetitions)
        
        # setup client and data provider
        self.client = Client('localhost', 2000)
        self.world = self.client.get_world()
        self.traffic_manager = self.client.get_trafficmanager(
                self.config.trafficmanager_port)
        self.traffic_manager.set_random_device_seed(
                self.config.trafficmanager_seed)

        CarlaDataProvider.set_client(self.client)
        CarlaDataProvider.set_world(self.world)
        CarlaDataProvider.set_traffic_manager_port(
                self.config.trafficmanager_port)

        signal.signal(signal.SIGINT, self._signal_handler)

        # set up blocking checks
        self.last_hero_positions = deque()
        self.max_positions_len = 60 
        self.blocking_distance = 3.0
        self.frame = 0
        self.target_idx = 0
        self.last_waypoint = 0
        

    def _signal_handler(self, signum, frame):
        if self.manager:
            self.manager.signal_handler(signum, frame)
        raise KeyboardInterrupt

    def reset(self, log=None):
        
        # load next RouteScenario
        num_configs = len(self.indexer._configs_list)
        rconfig = self.indexer.get(np.random.randint(num_configs))
        rconfig.agent = self.agent_instance

        
        self._load_world_and_scenario(rconfig)
        #self._get_hero_route(draw=True)
        self._get_hero_route()

        self.manager.start_system_time = time.time()
        self.manager.start_game_time = GameTime.get_time()
        self.manager._watchdog.start()
        self.manager._running = True

        self.agent_instance.reset()
        self.frame = 0
        self.last_hero_positions = deque()
        self.last_waypoint = 0

        if log is not None:
            self.env_log = {}
            self.env_log['town'] = rconfig.town
            self.env_log['name'] = rconfig.name
            self.env_log['total_waypoints'] = len(self.route_waypoints)
            self.env_log['last_waypoint'] = 0
            log['env'] = self.env_log


        return np.zeros(self.obs_dim)

    def _load_world_and_scenario(self, rconfig):

        # setup world and retrieve map
        self.world = self.client.load_world(rconfig.town)
        settings = self.world.get_settings()
        settings.fixed_delta_seconds = 1.0 / 20.0
        settings.synchronous_mode = True
        self.world.apply_settings(settings)
        self.world.reset_all_traffic_lights()
        self.traffic_manager = self.client.get_trafficmanager(
                self.config.trafficmanager_port)
        self.traffic_manager.set_synchronous_mode(True)

        # setup provider and tick to check correctness
        CarlaDataProvider.set_client(self.client)
        CarlaDataProvider.set_world(self.world)
        CarlaDataProvider.set_traffic_manager_port(
                self.config.trafficmanager_port)
        self.map = CarlaDataProvider.get_map()
        
        if CarlaDataProvider.is_sync_mode():
            self.world.tick()
        else:
            self.world.wait_for_tick()
        
        # setup scenario and scenario manager
        self.scenario = RouteScenario(
                self.world, rconfig, 
                criteria_enable=False, 
                env_config=self.config)
        self.manager.load_scenario(
                self.scenario, rconfig.agent,
                rconfig.repetition_index)

        self.hero_actor = CarlaDataProvider.get_hero_actor()
        if CarlaDataProvider.is_sync_mode():
            self.world.tick()
        else:
            self.world.wait_for_tick()

    def _get_hero_route(self):

        # retrieve new hero route
        self.route = CarlaDataProvider.get_ego_vehicle_route()

        route_locations = [route_elem[0] for route_elem in self.route]
        self.route_waypoints = [self.map.get_waypoint(loc) for loc in route_locations]
        self.route_transforms = [waypoint_to_vector(wp) for wp in self.route_waypoints]
        self.route_transforms = np.array(self.route_transforms)
        self.forward_vectors = [wp.transform.get_forward_vector() for wp in self.route_waypoints]
        self.forward_vectors = np.array( [[v.x, v.y, v.z] for v in self.forward_vectors])

        self.route_len = len(self.route_transforms)


    # convert action to vehicle control and tick scenario
    def step(self, action):
        if self.manager._running:
            timestamp = None
            snapshot = self.world.get_snapshot()
            if snapshot:
                timestamp = snapshot.timestamp
            if timestamp:
                obs, reward, done, info = self._tick(timestamp)
            self.frame += 1
            return obs, reward, done, info

        else:
            raise Exception
            return np.zeros(self.obs_dim), -10, True, {'running': False}

    def _tick(self, timestamp):

        self.manager._tick_scenario(timestamp) # ticks
        hero_transform = CarlaDataProvider.get_transform(self.hero_actor)

        # check blocked and timeout
        info = {}
        blocked_done = self._check_blocked(hero_transform)
        if blocked_done:
            info['blocked'] = True
        timeout_done = self.frame >= 6000
        if timeout_done:
            info['timeout'] = True

        # get target
        target_idx, distance_done = self._get_target(hero_transform) # idxs
        target_waypoint = self.route_waypoints[target_idx]
        if distance_done:
            info['too_far'] = True
                
        # get new state, reward, done, and update agent's cached reward for viz
        obs = self._get_observation(hero_transform, target_idx)
        
        #obs = self._get_waypoint_state(hero_transform, target_waypoint)
        reward_info = self._get_reward(hero_transform, target_waypoint, distance_done or blocked_done)
        self.agent_instance.cached_rinfo = reward_info
        self.agent_instance.make_visualization()

        # make visualizations
        draw_waypoints(self.world, [target_waypoint], color=(0,255,0), size=0.5)
        draw_arrow(self.world, hero_transform.location,
                target_waypoint.transform.location, color=(255,0,0), size=0.5)
        
        criteria = [blocked_done, timeout_done, distance_done]
        done = any(criteria)
        return obs, reward_info['reward'], done, info

    def _check_blocked(self, hero_transform):

        hero_vector = transform_to_vector(hero_transform)
        if self.frame > 60:
            if len(self.last_hero_positions) < self.max_positions_len:
                #self.last_hero_positions.append(hero_vector[:2])
                self.last_hero_positions.append(hero_transform)
            else:
                self.last_hero_positions.popleft()
                self.last_hero_positions.append(hero_vector[:2])
                start = self.last_hero_positions[0]
                end = self.last_hero_positions[-1]
                traveled = np.linalg.norm(end-start)
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
        y_cart, x_cart = heading_vector[:,1], heading_vector[:,0] # in carla coordinate system
        heading_angles = np.arctan2(y_cart, x_cart) * 180 / np.pi
        reachable_b = np.abs(heading_angles) < 100

        # aligned criteria - is the waypoint pointing the same direction we are?
        yaw_diffs = route_transforms[:, 4] - hero_transform_vec[4]
        yaw_diffs = (yaw_diffs + 180) % 360 - 180
        yaw_diffs = np.array([np.abs(yaw_diffs)]).flatten()
        aligned_b = yaw_diffs < 120 # some sharp right/left turns are > 90 degrees

        criteria = np.array([reachable_b, aligned_b]) # 2xN
        valid = np.prod(criteria, axis=0).flatten().astype(bool)
        valid_indices = np.arange(len(route_transforms))[valid]

        if len(valid_indices) == 0:
            return self.last_waypoint, True

        # retrieve target
        target_idx = self.last_waypoint + valid_indices[0]
        target = self.route_waypoints[target_idx]
        self.last_waypoint = self.last_waypoint + valid_indices[0]
        self.env_log['last_waypoint'] = int(self.last_waypoint)

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
        
        return target_idx, done
    
    
    def _get_observation(self, hero_transform, target_idx):

        # last 5 positions transformed into our frame
        history_size = 5
        if len(self.last_hero_positions) < history_size + 1:
            history = np.zeros(4*history_size)
        else:
            history = self.last_hero_positions[-(history_size+1):-1] # last position is current one
            positions = np.hstack([history[:,:3], np.ones((history_size,1)])
            world_to_hero = hero_transform.get_inverse_matrix()
            positions = np.matmul(world_to_hero, positions)

        # state per waypoint (x,y,z in agent frame + yaw diff) and agent velocity
        obs = np.zeros(self.obs_dim)

        max_len = 1e-9
        for i in range(self.nstate_waypoints):

            idx = min(len(self.route_waypoints)-1, target_idx + i)
            wpt = self.route_waypoints[idx]
            if i != 0:
                draw_waypoints(self.world, [wpt], color=(0,100,100), size=0.5)
            start, end = self.waypoint_state_dim*i, self.waypoint_state_dim*(i+1)
            obs[start:end] = self._get_waypoint_state(hero_transform, wpt)
            dist = np.linalg.norm(obs[start:start+3])
            max_len = max(max_len, dist)

        # norm distance and dyaw
        for i in range(self.nstate_waypoints):
            start, end = self.waypoint_state_dim*i, self.waypoint_state_dim*(i+1)
            obs[start:start+3] /= max_len
            obs[start+4] /= 180

        # velocity
        velocity = CarlaDataProvider.get_velocity(self.hero_actor)
        velocity = velocity * 3600 / 1000 # km/h
        norm_velocity = max(min(velocity, 80), 0) # clip to 0, 80
        norm_velocity = norm_velocity / 40 - 1 # squash to -1, 1
        obs[-2] = norm_velocity

        # completion
        completion = self.last_waypoint / self.route_len
        norm_completion = completion * 2 - 1
        obs[-1] = norm_completion
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

        # compute difference in yaws
        hyaw = hero_transform.rotation.yaw
        tyaw = target_waypoint.transform.rotation.yaw
        dyaw = sgn_angle_diff(hyaw, tyaw)
        
        state = np.array([x, y, z, dyaw])
        return state

    def _get_reward(self, hero_transform, target_waypoint, blocked_or_distance_done):
        hero = transform_to_vector(hero_transform)
        target = waypoint_to_vector(target_waypoint)

        # distance reward
        dist_max = (4**2 + self.config.hop_resolution**2)**0.5
        dist = min(np.linalg.norm(hero[:3] - target[:3]), dist_max)
        dist_reward = (dist/dist_max - 1)**2 - 1
        #dist_reward = 0 - min(dist/dist_max, 1)

        # rotation reward
        yaw_diff = (hero[4]-target[4]) % 360
        yaw_diff = yaw_diff if yaw_diff < 180 else 360 - yaw_diff
        yaw_max = 90
        yaw_frac = min(yaw_diff/yaw_max, 1)
        yaw_reward = -yaw_frac**2 + 1
        #yaw_reward = 1 - min(yaw_diff/yaw_max, 1)
        #yaw_reward = yaw_reward * 0.5

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

        reward = dist_reward + yaw_reward + vel_reward + route_reward
        reward_info = {
                'reward': reward, 
                'dist_reward': dist_reward,
                'yaw_reward': yaw_reward,
                'vel_reward': vel_reward,
                'route_reward': route_reward}
        return reward_info
        
    def cleanup(self):

        self.manager.stop_scenario(analyze=False)

        # Simulation still running and in synchronous mode?
        if self.manager and self.manager.get_running_status() and self.world:
            # Reset to asynchronous mode
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            self.world.apply_settings(settings)
            self.traffic_manager.set_synchronous_mode(False)

        if self.manager:
            self.manager.cleanup()

        CarlaDataProvider.cleanup()

        self.world = None
        self.map = None
        self.scenario = None
        self.traffic_manager = None

        self.hero_actor = None

        if self.agent_instance:
            # just clears sensor interface for resetting
            self.agent_instance.destroy() 

    def __del__(self):
        if hasattr(self, 'manager') and self.manager:
            del self.manager
        if hasattr(self, 'world') and self.world:
            del self.world
        if hasattr(self, 'scenario') and self.scenario:
            del self.scenario

    def render(self):
        pass

    def close(self):
        pass
