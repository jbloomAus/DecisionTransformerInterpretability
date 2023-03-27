import json
from dataclasses import dataclass, field
import argparse
import re
from typing import List

from minigrid.wrappers import OneHotPartialObsWrapper, RGBImgPartialObsWrapper

from .model import DecisionTransformer as DecisionTransformerLegacy
import torch as t

from src.models.trajectory_model import DecisionTransformer
from src.config import EnvironmentConfig, TransformerModelConfig, OfflineTrainConfig
from .offline_dataset import TrajectoryDataset


@dataclass
class DTArgs:
    exp_name: str = "Dev"
    d_model: int = 128
    trajectory_path: str = "trajectories/MiniGrid-Dynamic-Obstacles-8x8-v0c8c5dccc-b418-492e-bdf8-2c21256cd9f3.pkl"
    n_heads: int = 4
    d_mlp: int = 256
    n_layers: int = 2
    n_ctx: int = 3
    layer_norm: bool = False
    batch_size: int = 64
    train_epochs: int = 10
    test_epochs: int = 3
    learning_rate: float = 0.0001
    linear_time_embedding: bool = False
    pct_traj: float = 1
    weight_decay: float = 0.001
    seed: int = 1
    track: bool = True
    wandb_project_name: str = "DecisionTransformerInterpretability"
    wandb_entity: str = None
    test_frequency: int = 100
    test_batches: int = 10
    eval_frequency: int = 100
    eval_episodes: int = 10
    initial_rtg: List[float] = field(default_factory=lambda: [0.0, 1.0])
    prob_go_from_end: float = 0.1
    eval_max_time_steps: int = 1000
    cuda: bool = True


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Decision Transformer",
        description="Train a decision transformer on a trajectory dataset.",
        epilog="The last enemy that shall be defeated is death.")
    parser.add_argument("--exp_name", type=str, default="Dev")
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--trajectory_path", type=str)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--d_mlp", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--n_ctx", type=int, default=3)
    parser.add_argument("--layer_norm", type=bool, default=False)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--train_epochs", type=int, default=10)
    parser.add_argument("--test_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--linear_time_embedding", type=bool,
                        default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--pct_traj", type=float, default=1)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--track", type=bool, default=False)
    parser.add_argument("--wandb_project_name", type=str,
                        default="DecisionTransformerInterpretability")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--test_frequency", type=int, default=100)
    parser.add_argument("--eval_frequency", type=int, default=100)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--initial_rtg", action='append',
                        help='<Required> Set flag', required=False, default=[0, 1])
    parser.add_argument("--prob_go_from_end", type=float, default=0.1)
    parser.add_argument("--eval_max_time_steps", type=int, default=1000)
    parser.add_argument("--cuda", action=argparse.BooleanOptionalAction)
    parser.add_argument("--model_type", type=str,
                        default="decision_transformer")
    args = parser.parse_args()
    return args


def load_decision_transformer(model_path, env):
    if model_stored_in_legacy_format(model_path):
        state_dict = t.load(model_path)
        if "state_encoder.weight" in state_dict.keys():
            return load_legacy_decision_transformer(state_dict, env)

        # get number of layers from the state dict
        num_layers = max([int(re.findall(r'\d+', k)[0])
                          for k in state_dict.keys() if "transformer.blocks" in k]) + 1
        d_model = state_dict['reward_embedding.0.weight'].shape[0]
        d_mlp = state_dict['transformer.blocks.0.mlp.W_out'].shape[0]
        n_heads = state_dict['transformer.blocks.0.attn.W_O'].shape[0]
        max_timestep = state_dict['time_embedding.weight'].shape[0] - 1
        n_ctx = state_dict['transformer.pos_embed.W_pos'].shape[0]
        layer_norm = 'transformer.blocks.0.ln1.w' in state_dict

        if 'state_encoder.weight' in state_dict:
            # otherwise it would be a sequential and wouldn't have this
            state_embedding_type = 'grid'

        if state_dict['time_embedding.weight'].shape[1] == 1:
            time_embedding_type = "linear"
        else:
            time_embedding_type = "embedding"

        environment_config = EnvironmentConfig(
            env_id=env.unwrapped.spec.id,
            one_hot_obs=isinstance(env.observation_space,
                                   OneHotPartialObsWrapper),
            img_obs=isinstance(env.observation_space, RGBImgPartialObsWrapper),
            view_size=env.unwrapped.observation_space["image"].shape[0],
            fully_observed=False,
            capture_video=False,
            render_mode='rgb_array')

        transformer_config = TransformerModelConfig(
            d_model=d_model,
            n_heads=n_heads,
            d_mlp=d_mlp,
            n_layers=num_layers,
            n_ctx=n_ctx,
            layer_norm=layer_norm,
            time_embedding_type=time_embedding_type,
            state_embedding_type=state_embedding_type,
        )
    else:
        state_dict, trajectory_data_set, transformer_config, _ = load_model_data(
            model_path)

        if "state_encoder.weight" in state_dict.keys():
            return load_legacy_decision_transformer(state_dict, env)

        # now we can create the model
        # model = DecisionTransformer(
        #     EnvironmentConfig(env.__spec__),
        # )
        environment_config = EnvironmentConfig(
            env_id=trajectory_data_set.metadata['args']['env_id'],
            one_hot_obs=trajectory_data_set.observation_type == "one_hot",
            view_size=trajectory_data_set.metadata['args']['view_size'],
            fully_observed=False,
            capture_video=False,
            render_mode='rgb_array')

    model = DecisionTransformer(
        environment_config=environment_config,
        transformer_config=transformer_config
    )

    model.load_state_dict(state_dict)
    return model

# To maintain backwards compatibility with the old models.


def load_legacy_decision_transformer(state_dict, env):

    # get number of layers from the state dict
    num_layers = max([int(re.findall(r'\d+', k)[0])
                      for k in state_dict.keys() if "transformer.blocks" in k]) + 1
    d_model = state_dict['reward_embedding.0.weight'].shape[0]
    d_mlp = state_dict['transformer.blocks.0.mlp.W_out'].shape[0]
    n_heads = state_dict['transformer.blocks.0.attn.W_O'].shape[0]
    max_timestep = state_dict['time_embedding.weight'].shape[0] - 1
    n_ctx = state_dict['transformer.pos_embed.W_pos'].shape[0]
    layer_norm = 'transformer.blocks.0.ln1.w' in state_dict

    if 'state_encoder.weight' in state_dict:
        # otherwise it would be a sequential and wouldn't have this
        state_embedding_type = 'grid'

    if state_dict['time_embedding.weight'].shape[1] == 1:
        time_embedding_type = "linear"
    else:
        time_embedding_type = "learned"

    # now we can create the model
    model = DecisionTransformerLegacy(
        env=env,
        n_layers=num_layers,
        d_model=d_model,
        d_mlp=d_mlp,
        state_embedding_type=state_embedding_type,
        time_embedding_type=time_embedding_type,
        n_heads=n_heads,
        max_timestep=max_timestep,
        n_ctx=n_ctx,
        layer_norm=layer_norm
    )

    model.load_state_dict(state_dict)
    return model


def model_stored_in_legacy_format(model_path):
    model_info = t.load(model_path)
    return "model_state_dict" not in model_info


def load_model_data(model_path):
    model_info = t.load(model_path)
    state_dict = model_info["model_state_dict"]
    transformer_config = TransformerModelConfig(
        **json.loads(model_info["transformer_config"]))
    offline_config = OfflineTrainConfig(
        **json.loads(model_info["offline_config"]))

    trajectory_data_set = TrajectoryDataset(
        trajectory_path=offline_config.trajectory_path,
        max_len=transformer_config.n_ctx // 3,
        pct_traj=offline_config.pct_traj,
        prob_go_from_end=offline_config.prob_go_from_end,
        device=transformer_config.device,
    )

    return state_dict, trajectory_data_set, transformer_config, offline_config


def get_max_len_from_model_type(model_type: str, n_ctx: int):
    '''
    Ihe max len in timesteps is 3 for decision transformers
    and 2 for clone transformers since decision transformers
    have 3 tokens per timestep and clone transformers have 2.

    This is a map between timestep and tokens. We start with one
    for the most recent state/action and then add another
    timestep for every 3 tokens for decision transformers and
    every 2 tokens for clone transformers.
    '''
    assert model_type in ["decision_transformer", "clone_transformer"]
    if model_type == "decision_transformer":
        return 1 + n_ctx // 3
    else:
        return 1 + n_ctx // 2
