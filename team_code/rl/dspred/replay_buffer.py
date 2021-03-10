import numpy as np
from collections import deque
from itertools import islice

class ReplayBuffer():
    def __init__(self, buffer_size, batch_size, n):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.n = n
        self.states = deque()
        self.actions = deque()
        self.rewards = deque()
        self.dones = deque()
        self.infos = deque()
        self.td_errors = deque()

        self.bufs = [self.states, self.actions, self.rewards, self.dones, self.infos]
        self.gamma = 0.99
        self.discount = np.array([self.gamma**t for t in range(n+1)])

    def add_experience(self, state, action, reward, done, info):
        exp = [state, action, reward, done, info]
        if len(self.states) >= self.buffer_size:
            for buf in self.bufs:
                buf.popleft()
        for buf, data in zip(self.bufs, exp):
            buf.append(data)

    # centralize data saving here? 
    def save_latest(self):
        pass

    # keep imitation/driving score reward split in info?
    # return rewards and infos as a list
    def sample(self, t):
        state = self.states[t]
        action = self.actions[t]
        info = self.infos[t]

        end = min(t + self.n, len(self.states) - 1)
        dones = list(islice(self.dones, t, end+1))
        if True in dones:
            end = dones.index(True)
        done = self.dones[end]
        rewards = list(islice(self.rewards, t, end+1))
        reward = np.dot(rewards, self.discount[:len(rewards)])
        condition = (end == len(self.states)-1) or self.dones[end]
        next_state = self.states[end] if condition else self.states[end+1]
        info['next_action'] = self.actions[end] if condition else self.actions[end+1]

        return state, action, reward, done, next_state, info

    def __len__(self):
        return len(self.states)
