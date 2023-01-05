
import os 
from dataclasses import dataclass


@dataclass
class DTArgs:
    exp_name: str = os.path.basename(globals().get("__file__", "DT_implementation").rstrip(".py"))
    trajectory_path: str = "trajectories/MiniGrid-Dynamic-Obstacles-6x6-v0.pkl"
    d_model = 128
    n_heads = 2
    d_mlp = 248
    n_layers = 2
    learning_rate: float = 0.00025
    batch_size: int = 128
    batches: int = 100
    max_len: int = 100
    pct_traj: float = 1.0
    seed: int = 1
    track: bool = True
    wandb_project_name: str = "DecisionTransformerInterpretability"
    wandb_entity: str = None
    test_frequency: int = 10
    test_batches: int = 10
    eval_frequency: int = 10
    eval_episodes: int = 50 
