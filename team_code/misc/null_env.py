import gym
import numpy as np


class NullEnv(gym.Env):
    
    def __init__(self, obs_spec, act_spec):

        space_map = {'box': gym.spaces.Box} # support for other envs if needed

        space_type, low, high, dim, dtype = obs_spec
        self.observation_space = space_map[space_type](low=low, high=high, shape=dim, dtype=dtype)
        space_type, low, high, dim, dtype = act_spec
        self.action_space = space_map[space_type](low=low, high=high, shape=dim, dtype=dtype)

    def reset(self):
        return np.zeros(self.obs_dim)

    def step(self, action):
        return np.zeros(self.obs_dim), 0, True, {}

    def render(self):
        pass

    def close(self):
        pass

