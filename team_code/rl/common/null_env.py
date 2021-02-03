import gym
import numpy as np

class NullEnv(gym.Env):
    
    def __init__(self, obs_dim, action_dim, odtype, adtype):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        print('creating observation space')
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=obs_dim, dtype=odtype)
        print('creating action space')
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=action_dim, dtype=adtype)

    def reset(self):
        print('reset null')
        return np.zeros(self.obs_dim)

    def step(self, action):
        print('step null')
        return np.zeros(self.obs_dim), 0, True, {}

    def render(self):
        print('render null')
        pass

    def close(self):
        print('close null')
        pass

