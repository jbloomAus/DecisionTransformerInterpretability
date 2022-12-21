import os
import pickle
import time
from dataclasses import asdict, dataclass


class TrajectoryWriter():
    '''
    The trajectory writer is responsible for writing trajectories to a file.
    During each rollout phase, it will collect:
        - the observations
        - the actions
        - the rewards
        - the dones
        - the infos
    And store them in a set of lists, indexed by batch b and time t. 
    '''
    def __init__(self, path, args: dataclass):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.infos = []
        self.path = path
        self.args = args

    def accumulate_trajectory(self, next_obs, reward, done, action, info):
        self.observations.append(next_obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.infos.append(info)
    
    def write(self):

        data = {
            'observations': self.observations,
            'actions': self.actions,
            'rewards': self.rewards,
            'dones': self.dones,
            'infos': self.infos
        }

        metadata = {
            "args": asdict(self.args), # Args such as ppo args
            "time": time.time() # Time of writing
        }

        if not os.path.exists(os.path.dirname(self.path)):
            os.makedirs(os.path.dirname(self.path))

        with open(self.path, 'wb') as f:
            pickle.dump({
                'data': data,
                'metadata': metadata
            }, f)

        print(f"Trajectory written to {self.path}")


class TrajectoryReader():
    '''
    The trajectory reader is responsible for reading trajectories from a file.
    '''
    def __init__(self, path):
        self.path = path

    def read(self):
        with open(self.path, 'rb') as f:
            data = pickle.load(f)

        return data

        