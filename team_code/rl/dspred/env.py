import os, json
import gym
import numpy as np
from collections import deque
from itertools import islice

from rl.common.env_utils import *
from rl.common.base_env import BaseEnv
from rl.dspred.replay_buffer import ReplayBuffer

from leaderboard.utils.statistics_util import penalty_dict
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.traffic_events import TrafficEventType

from PIL import Image

class CarlaEnv(BaseEnv):

    def __init__(self, config, client, agent):
        super().__init__(config, client, agent)

        # RL params
        self.history_size = config.agent.history_size
        self.last_hero_transforms = deque()
        self.max_positions_len = 60 
        self.blocking_distance = 2.0
        self.target_idx = 0
        self.last_waypoint = 0

        self.num_infractions = 0
        self.buf = ReplayBuffer(config.agent.buffer_size, config.agent.batch_size)
        self.warmup_frames = 60

    def reset(self, log=None):
        # pass rconfig to hero agent reset method so it doesn't need 
        # environ variables to set save debug/image paths?

        # extract indexer? make a config parameter that chooses between
        # random rconfig selection or getting in order?
        rconfig = None
        if self.config.save_data:
            if self.indexer.peek():
                rconfig = self.indexer.next()
            else:
                return 'done'

        super().reset(log, rconfig)
        if self.config.save_data:
            save_root = self.all_config.save_root             
            ROUTE_NAME = os.environ.get('ROUTE_NAME', 0)
            repetition = 0
            for name in os.listdir(save_root):
                repetition += 1 if ROUTE_NAME in name else 0
            self.save_data_root = f'{save_root}/{ROUTE_NAME}_repetition_{repetition:02d}'
            os.makedirs(f'{self.save_data_root}/topdown')
            os.makedirs(f'{self.save_data_root}/measurements')

        return 'running'

    def step(self):
        # ticks the scenario and makes visual with new semantic bev image and cached info
        obs, reward, done, info = super().step() 
        if self.frame < self.warmup_frames:
            return (0, done)

        # cache things to make driving score compute faster?
        rclist = CarlaDataProvider.get_route_completion_list()
        reward = rclist[-1] - rclist[-2]
        iflist = CarlaDataProvider.get_infraction_list()
        if self.num_infractions < len(iflist): # new infraction
            self.num_infractions = len(iflist)
            itype = iflist[-1].get_type()

            # ignore stops for now
            if itype != TrafficEventType.STOP_INFRACTION and itype in penalty_dict.keys(): 
                base_penalty = 50
                penalty = base_penalty * (1 - penalty_dict[infraction.get_type()])
                reward = reward - penalty

        state = self.hero_agent.obs
        aim = self.hero_agent.aim
        self.buf.add_experience(state, aim, reward, done, info)

        if self.config.save_data:
            save_frame = self.frame - self.warmup_frames
            topdown, target = state
            Image.fromarray(topdown).save(f'{self.save_data_root}/topdown/{save_frame:06d}.png')

            data = {'x_tgt': float(target[0]),
                    'y_tgt': float(target[1]),
                    'x_aim': float(aim[0]),
                    'y_aim': float(aim[1]),
                    'reward': reward,
                    'done': int(done),
                    }
            with open(f'{self.save_data_root}/measurements/{save_frame:06d}.json', 'w') as f:
                json.dump(data, f, indent=4, sort_keys=False)

        return reward, done
