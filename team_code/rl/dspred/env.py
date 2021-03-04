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

        self.num_infractions = 0
        self.buffer = ReplayBuffer(
                config.agent.buffer_size, 
                config.agent.batch_size,
                config.agent.n)
        self.warmup_frames = 60

    def reset(self, log=None, rconfig=None):
        # pass rconfig to hero agent reset method so it doesn't need 
        # environ variables to set save debug/image paths?

        # extract indexer? make a config parameter that chooses between
        # random rconfig selection or getting in order?
        super().reset(log, rconfig)
        return 'running'

    def step(self):
        # ticks the scenario and makes visual with new semantic bev image and cached info
        obs, reward, done, info = super().step() 
        if self.frame < self.warmup_frames:
            return (0, done)

        route_completion = CarlaDataProvider.get_route_completion_list()
        reward = route_completion[-1] - route_completion[-2]
        infractions = CarlaDataProvider.get_infraction_list()
        if self.num_infractions < len(infractions): # new infraction
            self.num_infractions = len(infractions)
            itype = infractions[-1].get_type()

            # ignore stops for now
            if itype != TrafficEventType.STOP_INFRACTION and itype in penalty_dict.keys(): 
                penalty = 50 * (1 - penalty_dict[itype]) # 50 base penalty
                reward = reward - penalty

        state = self.hero_agent.state
        action = self.hero_agent.action
        self.buffer.add_experience(state, action, reward, done, info)

        return reward, done
