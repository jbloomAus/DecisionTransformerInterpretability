
from dataclasses import dataclass
import argparse

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
    n_ctx: int = 3
    layer_norm: bool = False
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
    initial_rtg: float = 1
    prob_go_from_end: float = 0.1
    eval_max_time_steps: int = 1000
    cuda: bool = True

def parse_args():
    parser = argparse.ArgumentParser(
        prog = "Decision Transformer",
        description="Train a decision transformer on a trajectory dataset.",
        epilog = "The last enemy that shall be defeated is death.") 
    parser.add_argument("--exp_name", type=str, default="Dev")
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--trajectory_path", type=str)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--d_mlp", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--batches", type=int, default=401)
    parser.add_argument("--n_ctx", type=int, default=3)
    parser.add_argument("--layer_norm", type=bool, default=False)
    parser.add_argument("--pct_traj", type=float, default=1)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--track", type=bool, default=False)
    parser.add_argument("--wandb_project_name", type=str, default="DecisionTransformerInterpretability")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--test_frequency", type=int, default=100)
    parser.add_argument("--test_batches", type=int, default=10)
    parser.add_argument("--eval_frequency", type=int, default=100)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--initial_rtg", type=float, default=1)
    parser.add_argument("--prob_go_from_end", type=float, default=0.1)
    parser.add_argument("--eval_max_time_steps", type=int, default=1000)
    parser.add_argument("--cuda", type=bool, default=True)
    args = parser.parse_args()
    return args
