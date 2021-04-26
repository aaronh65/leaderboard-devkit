import os, json
import gym
import numpy as np
from collections import deque
from itertools import islice

#from rl.common.env_utils import *
from dqn.src.env.base_env import BaseEnv
#from rl.dspred.replay_buffer import ReplayBuffer
#from rl.dspred.global_buffer import ReplayBuffer

from leaderboard.utils.statistics_util import penalty_dict, collision_types
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.traffic_events import TrafficEventType

from PIL import Image

class CarlaEnv(BaseEnv):

    def __init__(self, config, client, agent):
        super().__init__(config, client, agent)

        self.warmup_frames = 20
        self.blocked_time = 5
        self.blocked_distance = 1.0

    def reset(self):

        status = super().reset()
        if status == 'done':
            return status

        self.last_collision_frame = -float('inf')
        self.hero_history = deque()
        self.itype = None
        self.num_infractions = 0

        #for step in range(self.warmup_frames):
        #    obs, reward, done, info = super().step() 
        #    if done:
        #        status = super().reset()
        return status

    
    def step(self):
        _, _, done, info = super().step() 

        # driving reward
        reward = self.compute_reward(info)
        done = done or self.check_blocked(info)
        
        if self.config.save_data:
            self.save_data(reward, done, info)
        #else:
        #    ReplayBuffer.add_env_data(reward, done, info)

        if self.econfig.short_stop:
            done = done or self.frame > 200

        return reward, done, info

    def rollout(self, num_steps=None, num_episodes=None, complete=None):
        assert num_steps != None or num_episodes != None or complete != None
        status = 'running'
        if complete != None:
            #print('rolling out until completion')
            while status != 'done':
                reward, done, info = self.step()
                if done:
                    self.cleanup()
                    status = self.reset()
        elif num_episodes != None:
            counter=0
            while counter < num_episodes and status != 'done':
                reward, done, info = self.step()
                if done:
                    self.cleanup()
                    status = self.reset()
                    counter += 1
        elif num_steps != None:
            counter=0
            while counter < num_steps and status != 'done':
                reward, done, info = self.step()
                counter += 1
                if done:
                    self.cleanup()
                    status = self.reset()

    def compute_reward(self, info, threshold=11):

        # PULL INFRACTION CHECK INTO BASE ENV
        infractions = CarlaDataProvider.get_infraction_list()
        if self.num_infractions < len(infractions): # new infraction
            self.num_infractions = len(infractions)
            self.itype = infractions[-1].get_type()
            print(f'{self.itype} at frame {self.frame}')
            info['infraction'] = infractions[-1]
        else:
            self.itype = None
        if self.itype != TrafficEventType.STOP_INFRACTION and self.itype in penalty_dict.keys():
            penalty = 100 * (1 - penalty_dict[self.itype]) # 50 base penalty
        else:
            penalty = 0

        # route reward
        route_completion = CarlaDataProvider.get_route_completion_list()
        route_reward = (route_completion[-1] - route_completion[-2])

        # imitation reward - take out this computation?
        # 0 reward if either point is > 2 meters away
        points_student = self.hero_agent.tick_data['points_map']
        points_expert = self.hero_agent.tick_data['points_expert']
        delta = np.linalg.norm(points_student[:2] - points_expert[:2], axis=1) # (2,1)
        imitation_reward = max((threshold - np.amax(delta))/threshold, 0)

        info['penalty'] = penalty
        info['imitation_reward'] = imitation_reward
        info['route_reward'] = route_reward

        return imitation_reward - penalty

    def save_data(self, reward, done, info):
        data = info
        x,y = self.hero_agent.tick_data['target']
        data['x_tgt'] = x
        data['y_tgt'] = y
        data['done'] = int(done)
        data['steer'] = self.hero_agent.control.steer
        data['throttle'] = self.hero_agent.control.throttle
        data['brake'] = self.hero_agent.control.brake
        if 'infraction' not in data.keys():
            data['infraction'] = 'none'
        else:
            data['infraction'] = str(data['infraction'].get_type())

        save_path = self.hero_agent.save_path / 'measurements'
        (save_path / f'{self.hero_agent.step:06d}.json').write_text(str(data))


    def check_blocked(self, info):

        if 'infraction' in info.keys() and info['infraction'].get_type() in collision_types:
            self.last_collision_frame = self.frame
            print('collision at frame', self.last_collision_frame)

        location = CarlaDataProvider.get_transform(self.hero_actor).location
        x, y = location.x, location.y
        if len(self.hero_history) < 20*self.blocked_time:
            self.hero_history.appendleft((x,y))
        else:
            self.hero_history.pop() # take out the oldest location
            self.hero_history.appendleft((x,y))

        # check for the 15 seconds after collision
        done = False
        if self.frame - self.last_collision_frame < 300 :
            x0, y0 = self.hero_history[0]
            x1, y1 = self.hero_history[-1]
            norm = ((x1-x0)**2+(y1-y0)**2)**0.5
            done = norm < self.blocked_distance
        return done
