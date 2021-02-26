import os, time, signal
import gym
import numpy as np

from leaderboard.scenarios.scenario_manager import ScenarioManager
from leaderboard.scenarios.route_scenario import RouteScenario
from leaderboard.utils.route_indexer import RouteIndexer
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.timer import GameTime

class BaseEnv(gym.Env):

    def __init__(self, config, client, agent):
        super(BaseEnv, self).__init__()

        self.all_config = config
        self.config = config.env
        self.rconfig = None
        self.client = client
        self.hero_agent = agent # the user-defined agent class
        self.hero_actor = None  # the actual CARLA actor

        # basic attributes + data provider setup
        self.world = self.client.get_world()
        self.traffic_manager = self.client.get_trafficmanager(
                self.config.trafficmanager_port)
        self.traffic_manager.set_random_device_seed(
                self.config.trafficmanager_seed)

        CarlaDataProvider.set_client(self.client)
        CarlaDataProvider.set_world(self.world)
        CarlaDataProvider.set_traffic_manager_port(
                self.config.trafficmanager_port)

        # leaderboard objects
        data_dir = f'{config.project_root}/leaderboard/data'
        self.indexer = RouteIndexer(
                f'{data_dir}/{self.config.routes}',
                f'{data_dir}/{self.config.scenarios}', 
                self.config.repetitions)
        self.num_routes = len(self.indexer._configs_list)
        self.manager = ScenarioManager(60, False) # 60s timeout?
        self.scenario = None

        # forwards SIGINTs to signal handler
        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, signum, frame):
        if self.manager:
            self.manager.signal_handler(signum, frame)
        raise KeyboardInterrupt

    def reset(self, log=None, rconfig=None):
        if rconfig is None:
            idx = np.random.randint(self.num_routes)
            rconfig = self.indexer.get(idx)
        rconfig.agent = self.hero_agent
        self._load_world_and_scenario(rconfig)


        self.frame = 0

        self.manager.start_system_time = time.time()
        self.manager.start_game_time = GameTime.get_time()
        self.manager._watchdog.start()
        self.manager._running = True

        if log is not None:
            self.env_log = {}
            self.env_log['town'] = rconfig.town
            self.env_log['name'] = rconfig.name
            log['env'] = self.env_log

        route_num = int(rconfig.name.split('_')[-1])
        route_name = f'route_{route_num:02d}'
        os.environ['ROUTE_NAME'] = route_name

        self.rconfig = rconfig
        self.hero_agent.reset()
        return []

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
                criteria_enable=True, 
                env_config=self.config)
        self.manager.load_scenario(
                self.scenario, rconfig.agent,
                rconfig.repetition_index)

        self.hero_actor = CarlaDataProvider.get_hero_actor()
        if CarlaDataProvider.is_sync_mode():
            self.world.tick()
        else:
            self.world.wait_for_tick()

    def step(self, action=None):
        if self.manager._running:
            timestamp = None
            snapshot = self.world.get_snapshot()
            if snapshot:
                timestamp = snapshot.timestamp
            if timestamp:
                self.manager._tick_scenario(timestamp)
            self.frame += 1
            done = not self.manager._running
            return [], -9999, done, {}
        else:
            print('manager not running')
            return [], -9999, True, {}

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

        if self.hero_agent:
            # just clears sensor interface for resetting
            self.hero_agent.destroy() 

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
