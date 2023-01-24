import os
import pickle
import time
import numpy as np
from typing import Dict
from dataclasses import asdict, dataclass
import wandb
import re
import torch as t
from typeguard import typechecked
from .decision_transformer.model import DecisionTransformer

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

    def tag_terminated_trajectories(self):
        '''
        Tag the last trajectory in each batch as done.
        This is needed when an episode in a minibatch is ended because the
        timesteps limit has been reached but the episode may not have been truncated
        or ended in the environment.

        I don't love this solution, but it will do for now.
        '''
        n_envs = len(self.dones[-1])
        for i in range(n_envs):
            self.truncated[-1][i] = True
    
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


def load_decision_transformer(model_path, env):

    state_dict = t.load(model_path)

    # get number of layers from the state dict
    num_layers = max([int(re.findall(r'\d+', k)[0]) for k in state_dict.keys() if "transformer.blocks" in k]) + 1
    d_model = state_dict['reward_embedding.0.weight'].shape[0]
    d_mlp = state_dict['transformer.blocks.0.mlp.W_out'].shape[0]
    n_heads = state_dict['transformer.blocks.0.attn.W_O'].shape[0]
    max_timestep = state_dict['time_embedding.weight'].shape[0] - 1
    n_ctx = state_dict['transformer.pos_embed.W_pos'].shape[0]
    layer_norm = 'transformer.blocks.0.ln1.w' in state_dict

    if 'state_encoder.weight' in state_dict:
        state_embedding_type = 'grid' # otherwise it would be a sequential and wouldn't have this 
        
    # now we can create the model 
    model = DecisionTransformer(
        env = env,
        n_layers = num_layers,
        d_model = d_model,
        d_mlp = d_mlp,
        state_embedding_type = state_embedding_type,
        n_heads = n_heads,
        max_timestep = max_timestep,
        n_ctx = n_ctx,
        layer_norm = layer_norm
    )

    model.load_state_dict(state_dict)
    return model
