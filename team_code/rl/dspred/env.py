import signal
import time, os
import gym
import numpy as np
import itertools
from collections import deque
from itertools import islice

from rl.common.env_utils import *
from rl.common.base_env import BaseEnv
from rl.dspred.replay_buffer import ReplayBuffer

from leaderboard.utils.statistics_util import penalty_dict
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.traffic_events import TrafficEventType


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

    def reset(self, log=None):
        super().reset(log)
        
        return []

    def step(self):
        # ticks the scenario and makes visual with new semantic bev image and cached info
        obs, reward, done, info = super().step() 
        if self.frame <= 60:
            return (0, done)

        # cache things to make driving score compute faster?
        rclist = CarlaDataProvider.get_route_completion_list()
        reward = rclist[-1] - rclist[-2]
        iflist = CarlaDataProvider.get_infraction_list()
        if self.num_infractions < len(iflist): # new infraction
            self.num_infractions = len(iflist)
            infraction = iflist[-1]
            if infraction.get_type() != TrafficEventType.STOP_INFRACTION: # ignore for now
                base_penalty = 50
                penalty = base_penalty * (1 - penalty_dict[infraction])
                reward = reward - penalty

        state = self.hero_agent.obs
        action = self.hero_agent.aim
        self.buf.add_experience(state, action, reward, done, info)
        return (reward, done) 
