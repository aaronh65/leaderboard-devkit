import numpy as np
from collections import deque

class ReplayBuffer():
    def __init__(self, n, batch_size):
        self.n = n
        self.batch_size = batch_size
        self.states = deque()
        self.actions = deque()
        self.rewards = deque()
        self.dones = deque()
        self.infos = deque()
        self.td_errors = deque()

        self.bufs = [self.states, self.actions, self.rewards, self.dones, self.infos]

    def add_experience(self, state, action, reward, done, info):
        exp = [state, action, reward, done, info]
        if len(self.states) >= self.n:
            for buf in self.bufs:
                buf.popleft()
        for buf, data in zip(self.bufs, exp):
            buf.append(data)

    def sample(self):
        indices = np.random.randint(len(self.states)-1, size=self.batch_size)
        batch = []
        for t in indices:
            exp = []
            for buf in self.bufs:
                exp.append(buf[t])
            next_state = self.states[t+1] if not self.dones[t] else self.states[t]
            exp.insert(3, next_state)
            batch.append(exp)
        return batch
