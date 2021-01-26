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

        self.observation_space = gym.spaces.Box(
                low=-1, high=1, shape=(6,), 
                dtype=np.float32)
        self.action_space = gym.spaces.Box(
                low=-1, high=1, shape=(3,), 
                dtype=np.float32)

        self.config = config.env
        self.agent_instance = agent
        self.manager = ScenarioManager(60, False)
        self.scenario = None
        self.hero_actor = None
        self.frame = 0

        # set up blocking checks
        self.last_hero_positions = deque()
        self.max_positions_len = 60 
        self.blocking_distance = 3.0
        self.target_idx = 0

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

    def _signal_handler(self, signum, frame):
        if self.manager:
            self.manager.signal_handler(signum, frame)
        raise KeyboardInterrupt

    def reset(self, log=None):
        
        # load next RouteScenario
        num_configs = len(self.indexer._configs_list)
        rconfig = self.indexer.get(np.random.randint(num_configs))
        #rconfig = self.indexer.get(0)
        rconfig.agent = self.agent_instance

        if log is not None:
            env_log = {}
            env_log['town'] = rconfig.town
            env_log['name'] = rconfig.name
            log['env'] = env_log

        self._load_world_and_scenario(rconfig)
        self._get_hero_route(draw=True)

        self.manager.start_system_time = time.time()
        self.manager.start_game_time = GameTime.get_time()
        self.manager._watchdog.start()
        self.manager._running = True

        self.agent_instance.reset()
        self.frame = 0
        self.last_hero_positions = deque()

        return np.zeros(6)

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
            return np.zeros(6), 0, True, {'running': False}

    def _tick(self, timestamp):

        self.manager._tick_scenario(timestamp) # ticks
        hero_transform = CarlaDataProvider.get_transform(self.hero_actor)

        # check if blocked
        if self._check_blocked(hero_transform):
            return np.zeros(6), 0, True, {'blocked': True}
        
        # get target
        idxs = self._get_target_idxs(hero_transform, num_targets=5) # idxs
        target_idx = idxs[self.target_idx]
        if len(idxs) < self.target_idx + 1:
            return np.zeros(6), 0, True, {'no_targets': True}
        target_waypoint = self.route_waypoints[target_idx]
        
        # get new state, reward, done, and update agent's cached reward for viz
        obs = self._get_observation(hero_transform, target_waypoint)
        reward_info, done = self._get_reward(hero_transform, target_waypoint)

        # refine the done flag
        # the distance metric only applies for waypoints in the same road/section/lane
        if target_idx > 0:
            previous_waypoint = self.route_waypoints[target_idx-1]
            done = done and previous_waypoint.road_id == target_waypoint.road_id
            done = done and previous_waypoint.lane_id == target_waypoint.lane_id
            done = done and previous_waypoint.section_id == target_waypoint.section_id

        # make visualizations
        draw_waypoints(self.world, [target_waypoint], color=(0,255,0), size=0.5)
        draw_arrow(self.world, hero_transform.location,
                target_waypoint.transform.location, color=(255,0,0), size=0.5)

        self.agent_instance.cached_rinfo = reward_info
        self.agent_instance.make_visualization()

        return obs, reward_info['reward'], done, {}

    def _check_blocked(self, hero_transform):

        hero_vector = transform_to_vector(hero_transform)
        if self.frame > 60:
            if len(self.last_hero_positions) < self.max_positions_len:
                self.last_hero_positions.append(hero_vector[:2])
            else:
                self.last_hero_positions.popleft()
                self.last_hero_positions.append(hero_vector[:2])
                start = self.last_hero_positions[0]
                end = self.last_hero_positions[-1]
                traveled = np.linalg.norm(end-start)
                if traveled < self.blocking_distance:
                    return True
                    
        return False

    def _get_target_idxs(self, hero_transform, num_targets=1):

        # distance criteria
        hero_transform_vec = transform_to_vector(hero_transform)
        hero2pt = self.route_transforms[:,:3] - hero_transform_vec[:3]
        dist2pt = np.linalg.norm(hero2pt, axis=1)
        indices = np.argsort(dist2pt).flatten()

        hero_fvec = hero_transform.get_forward_vector()
        hero_fvec = np.array([cvector_to_array(hero_fvec)]).T # 3x1

        # reachable criteria - does it take a >90 deg turn to get to waypoint?
        reachable = np.matmul(hero2pt, hero_fvec)
        reachable = reachable > 0

        # aligned criteria - is the waypoint pointing the same direction we are?
        yaw_diffs = self.route_transforms[:, 4] - hero_transform_vec[4]
        yaw_diffs = (yaw_diffs + 180) % 360 - 180
        yaw_diffs = np.array([np.abs(yaw_diffs)]).T
        aligned = yaw_diffs < 120 # some sharp right/left turns are > 90 degrees
        
        #maybe_aligned = np.matmul(fvectors, hero_fvec)
        #aligned = maybe_aligned > 0

        criteria = np.hstack([reachable, aligned]) # Nx2
        valid = np.prod(criteria, axis=1).flatten()
        valid = valid[indices] # reorder by distance criteria
        valid_indices = indices[valid] # slice out valid indices

        targets = valid_indices[:num_targets]
        return targets
    
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
                self.world, 
                rconfig, 
                criteria_enable=False, 
                env_config=self.config)
        self.manager.load_scenario(
                self.scenario, 
                rconfig.agent,
                rconfig.repetition_index)

        self.hero_actor = CarlaDataProvider.get_hero_actor()
        if CarlaDataProvider.is_sync_mode():
            self.world.tick()
        else:
            self.world.wait_for_tick()

    def _get_hero_route(self, draw=False):

        # retrieve new hero route
        self.route = CarlaDataProvider.get_ego_vehicle_route()

        # fill in the gaps
        

        route_locations = [route_elem[0] for route_elem in self.route]
        self.route_waypoints = [
                self.map.get_waypoint(loc) for loc in route_locations]
        self.route_transforms = [
                waypoint_to_vector(wp) for wp in self.route_waypoints]
        self.route_transforms = np.array(self.route_transforms)
        forward_vectors = [wp.transform.get_forward_vector() for wp in self.route_waypoints]
        self.forward_vectors = np.array( [[v.x, v.y, v.z] for v in forward_vectors])

        if draw:
            draw_waypoints(self.world, self.route_waypoints, color=(0,0,255), life_time=9999)

    def _get_observation(self, hero_transform, target_waypoint, verbose=False):

        # transform target from world frame to hero frame
        target = waypoint_to_vector(target_waypoint)
        x,y,z = target[:3]
        target_location = np.array([[x,y,z,1]]).T # 4x1
        world_to_hero = hero_transform.get_inverse_matrix() # 4x4
        target_in_hero = np.matmul(world_to_hero, target_location).flatten()
        target_in_hero = target_in_hero[:3]

        # compute heading angle from hero to target 
        x,y,z = target_in_hero
        dist = (x**2 + y**2)**0.5
        heading = np.arctan2(y,x) * 180/np.pi if dist > 0.1 else 0 # degrees

        # compute difference in yaws
        hyaw = hero_transform.rotation.yaw
        tyaw = target_waypoint.transform.rotation.yaw
        dyaw = sgn_angle_diff(hyaw, tyaw)

        # velocity
        velocity = CarlaDataProvider.get_velocity(self.hero_actor)
        velocity = velocity * 3600 / 1000 # km/h
        
        # normalize
        norm_target_in_hero = target_in_hero / np.linalg.norm(target_in_hero) # -1 to 1
        x,y,z = norm_target_in_hero
        norm_heading = heading / 180 # -1 to 1
        norm_dyaw = dyaw / 180 # -1 to 1
        norm_velocity = max(min(velocity, 80), 0) # clip to 0, 80
        norm_velocity = norm_velocity / 40 - 1 # squash to -1, 1

        obs = np.array([x, y, z, norm_heading, norm_dyaw, norm_velocity])

        if verbose:
            print()
            print(hero_transform)
            print(target_waypoint.transform)
            print(f'target in hero frame = {target_in_hero}')
            print(f'heading = {heading} deg')
            print(f'dyaw = {dyaw} deg')
            print(f'velocity = {velocity} km/h')
            print(obs)

        return obs

    def _get_reward(self, hero_transform, target_waypoint):
        hero = transform_to_vector(hero_transform)
        target = waypoint_to_vector(target_waypoint)

        # distance reward
        dist = np.linalg.norm(hero[:3] - target[:3])
        dist_max = 5
        dist_reward = 1 - min(dist/dist_max, 1)

        # rotation reward
        yaw_diff = (hero[4]-target[4]) % 360
        yaw_diff = yaw_diff if yaw_diff < 180 else 360 - yaw_diff
        yaw_max = 90
        yaw_reward = 1 - min(yaw_diff/yaw_max, 1)
        yaw_reward = yaw_reward * 0.5

        # speed reward
        hvel = CarlaDataProvider.get_velocity(self.hero_actor) # m/s
        hvel = hvel * 3600 / 1000 # km/h
        tvel = 40 # km/h
        vel_diff = abs(hvel-tvel)
        vel_reward = 1 - min(vel_diff/tvel, 1)

        reward = dist_reward + yaw_reward + vel_reward
        done = dist > dist_max

        #print(f'abs({hero[4]} - {target[4]}) = {yaw_diff}')
        #print(f'reward: {dist_reward:.2f} + {yaw_reward:.2f} + {vel_reward:.2f} = {reward:.2f}')
        #print(f'done: {done}')
        #print()
        reward_info = {
                'reward': reward, 
                'dist_reward': dist_reward,
                'yaw_reward': yaw_reward,
                'vel_reward': vel_reward}

        return reward_info, done
        

    def render(self):
        pass

    def close(self):
        pass
