import os
import pickle
import time
import numpy as np
from typing import Dict
from dataclasses import asdict, dataclass
import wandb

from typeguard import typechecked

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
        self.truncated = []
        self.infos = []
        self.path = path
        self.args = args
    
    @typechecked
    def accumulate_trajectory(self, next_obs: np.ndarray, reward: np.ndarray, done: np.ndarray, truncated: np.ndarray, action: np.ndarray, info: Dict):
        self.observations.append(next_obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.truncated.append(truncated)
        self.infos.append(info)
    
    def write(self, upload_to_wandb: bool = False):

        data = {
            'observations': np.array(self.observations, dtype = np.float64),
            'actions': np.array(self.actions, dtype = np.int64),
            'rewards': np.array(self.rewards, dtype = np.float64),
            'dones': np.array(self.dones, dtype = bool),
            'truncated': np.array(self.truncated, dtype = bool),
            'infos': np.array(self.infos, dtype = object)
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

        if upload_to_wandb:
            artifact = wandb.Artifact(self.path.split("/")[-1], type = "trajectory")
            artifact.add_file(self.path)
            wandb.log_artifact(artifact)

        print(f"Trajectory written to {self.path}")


