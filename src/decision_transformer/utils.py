
from dataclasses import dataclass


@dataclass
class DTArgs:
    exp_name: str = "Dev"
    d_model: int = 128
    trajectory_path: str = "trajectories/MiniGrid-Dynamic-Obstacles-8x8-v0c8c5dccc-b418-492e-bdf8-2c21256cd9f3.pkl"
    n_heads: int = 4
    d_mlp: int = 256
    n_layers: int = 2
    learning_rate: float = 0.0001
    batch_size: int = 64
    batches: int = 401
    max_len: int = 10
    layer_norm: bool = True
    pct_traj: float = 1
    weight_decay: float = 0.001
    seed: int = 1
    track: bool = True
    wandb_project_name: str = "DecisionTransformerInterpretability"
    wandb_entity: str = None
    test_frequency: int = 100
    test_batches: int = 10
    eval_frequency: int = 100
    eval_episodes: int = 1
    initial_rtg: float = 1
    eval_max_time_steps: int = 1000
