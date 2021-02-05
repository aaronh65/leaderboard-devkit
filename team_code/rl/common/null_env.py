import gym
import numpy as np


class NullEnv(gym.Env):
    
    def __init__(self, obs_dim, action_dim, odtype, adtype):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=obs_dim, dtype=odtype)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=action_dim, dtype=adtype)

    def create_obs_space(self, space_type, low, high, dim, dtype):
        space_map = {'box': gym.spaces.Box, 'discrete': gym.spaces.Discrete}
        space = space_map[space_type]
        self.observation_space = space(low=low, high=high, shape=dim, dtype=dtype)

    def create_action_space(self, space_type, low, high, dim, dtype):
        space_map = {'box': gym.spaces.Box, 'discrete': gym.spaces.Discrete}
        space = space_map[space_type]
        self.action_space = space(low=low, high=high, shape=dim, dtype=dtype)

    def reset(self):
        return np.zeros(self.obs_dim)

    def step(self, action):
        return np.zeros(self.obs_dim), 0, True, {}

    def render(self):
        pass

    def close(self):
        pass

