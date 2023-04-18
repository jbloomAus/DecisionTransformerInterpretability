import dataclasses
import gzip
import json
import lzma
import os
import pickle
import time
from typing import Dict

import numpy as np
from typeguard import typechecked

import wandb
from src.config import ConfigJsonEncoder


class TrajectoryWriter:
    """
    The trajectory writer is responsible for writing trajectories to a file.
    During each rollout phase, it will collect:
        - the observations
        - the actions
        - the rewards
        - the dones
        - the infos
    And store them in a set of lists, indexed by batch b and time t.
    """

    def __init__(
        self,
        path,
        run_config,
        environment_config,
        online_config,
        model_config=None,
    ):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.truncated = []
        self.infos = []
        self.path = path

        args = (
            run_config.__dict__
            | environment_config.__dict__
            | online_config.__dict__
        )
        if model_config is not None:
            args = args | model_config.__dict__

        self.args = args

    @typechecked
    def accumulate_trajectory(
        self,
        next_obs: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        truncated: np.ndarray,
        action: np.ndarray,
        info: Dict,
    ):
        self.observations.append(next_obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.truncated.append(truncated)
        self.infos.append(info)

    def tag_terminated_trajectories(self):
        """
        Tag the last trajectory in each batch as done.
        This is needed when an episode in a minibatch is ended because the
        timesteps limit has been reached but the episode may not have been truncated
        or ended in the environment.

        I don't love this solution, but it will do for now.
        """
        n_envs = len(self.dones[-1])
        for i in range(n_envs):
            self.truncated[-1][i] = True

    def write(self, upload_to_wandb: bool = False):
        data = {
            "observations": np.array(self.observations, dtype=np.float),
            "actions": np.array(self.actions, dtype=np.int64),
            "rewards": np.array(self.rewards, dtype=np.float),
            "dones": np.array(self.dones, dtype=bool),
            "truncated": np.array(self.truncated, dtype=bool),
            "infos": np.array(self.infos, dtype=object),
        }
        if dataclasses.is_dataclass(self.args):
            metadata = {
                # Args such as ppo args
                "args": json.dumps(self.args, cls=ConfigJsonEncoder),
                "time": time.time(),  # Time of writing
            }
        else:
            metadata = {
                "args": self.args,  # Args such as ppo args
                "time": time.time(),  # Time of writing
            }

        if not os.path.exists(os.path.dirname(self.path)):
            os.makedirs(os.path.dirname(self.path))

        # use lzma to compress the file
        if self.path.endswith(".xz"):
            print(f"Writing to {self.path}, using lzma compression")
            with lzma.open(self.path, "wb") as f:
                pickle.dump({"data": data, "metadata": metadata}, f)
        elif self.path.endswith(".gz"):
            print(f"Writing to {self.path}, using gzip compression")
            with gzip.open(self.path, "wb") as f:
                pickle.dump({"data": data, "metadata": metadata}, f)
        else:
            print(f"Writing to {self.path}")
            with open(self.path, "wb") as f:
                pickle.dump({"data": data, "metadata": metadata}, f)

        if upload_to_wandb:
            artifact = wandb.Artifact(
                self.path.split("/")[-1], type="trajectory"
            )
            artifact.add_file(self.path)
            wandb.log_artifact(artifact)

        print(f"Trajectory written to {self.path}")
