import numpy as np
from collections import deque
from itertools import islice

class ReplayBuffer(object):


    buffer_size = -1
    batch_size = -1
    n = -1
    gamma = 0.99
    discount = None

    states = deque()
    actions = deque()
    rewards = deque()
    dones = deque()
    infos = deque()
    td_errors = deque()

    bufs = None

    @staticmethod
    def setup(buffer_size, batch_size, n):
        ReplayBuffer.buffer_size = buffer_size
        ReplayBuffer.batch_size = batch_size
        ReplayBuffer.n = n
        ReplayBuffer.bufs = [ReplayBuffer.states, ReplayBuffer.actions, ReplayBuffer.rewards, ReplayBuffer.dones, ReplayBuffer.infos]
        ReplayBuffer.discount = np.array([ReplayBuffer.gamma**t for t in range(n+1)])

    @staticmethod
    def add_experience(state, action, reward, done, info):
        exp = [state, action, reward, done, info]
        if len(ReplayBuffer.states) >= self.buffer_size:
            for buf in ReplayBuffer.bufs:
                buf.popleft()
        for buf, data in zip(ReplayBuffer.bufs, exp):
            buf.append(data)

    # centralize data saving here? 
    @staticmethod
    def save_latest():
        pass

    # keep imitation/driving score reward split in info?
    # return rewards and infos as a list
    @staticmethod
    def sample(t):

        t = t % len(ReplayBuffer.states)

        state = ReplayBuffer.states[t]
        action = ReplayBuffer.actions[t]
        info = ReplayBuffer.infos[t]

        end = min(t + ReplayBuffer.n, len(self.states) - 1)
        dones = list(islice(ReplayBuffer.dones, t, end+1))
        if True in dones:
            end = dones.index(True)
        done = ReplayBuffer.dones[end]
        rewards = list(islice(ReplayBuffer.rewards, t, end+1))
        reward = np.dot(rewards, ReplayBuffer.discount[:len(rewards)])
        condition = (end == len(ReplayBuffer.states)-1) or self.dones[end]
        next_state = ReplayBuffer.states[end] if condition else self.states[end+1]
        info['next_action'] = ReplayBuffer.actions[end] if condition else self.actions[end+1]

        return state, action, reward, done, next_state, info

    @staticmethod
    def __len__():
        return ReplayBuffer.buffer_size
