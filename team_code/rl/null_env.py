import gym
import numpy as np

class NullEnv(gym.Env):
    
    def __init__(self, obs_dim, action_dim):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(obs_dim,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(action_dim,), dtype=np.float32)

    def reset(self):
        return np.zeros(self.obs_dim)

    def step(self, action):
        return np.zeros(self.obs_dim), 0, True, {}

    def render(self):
        pass

    def close(self):
        pass

