import torch as t
import warnings

from environments import make_env

from decision_transformer.utils import DTArgs
from decision_transformer.decision_transformer import DecisionTransformer
from decision_transformer.offline_dataset import TrajectoryLoader
from decision_transformer.train import train, test, evaluate_dt_agent


warnings.filterwarnings("ignore", category=DeprecationWarning)
device = t.device("cuda" if t.cuda.is_available() else "cpu")

if __name__ == "__main__":

    args = DTArgs()

    if args.trajectory_path is None:
        raise ValueError("Must specify a trajectory path.")

    trajectory_data_set = TrajectoryLoader(
        args.trajectory_path, pct_traj=args.pct_traj, device=device)

    # make an environment 
    env_id = trajectory_data_set.metadata['args']['env_id']
    env = make_env(env_id, seed = 0, idx = 0, capture_video=False, run_name = "dev", fully_observed=False)
    env = env()

    # make a decision transformer
    dt = DecisionTransformer(
        env = env, 
        d_model = args.d_model,
        n_heads = args.n_heads,
        d_mlp = args.d_mlp,
        n_layers = args.n_layers,
        state_embedding_type="grid", # hard-coded for now to minigrid.
        max_timestep=trajectory_data_set.metadata.get("args").get("max_steps") # Our DT must have a context window large enough
    )

    dt = train(dt, trajectory_data_set, env, device=device, max_len=args.max_len,
               batches=args.batches, lr=args.learning_rate, batch_size=args.batch_size)

    loss, accuracy = test(dt, trajectory_data_set, make_env,
                          device="cpu", max_len=args.max_len, batch_size=args.batch_size)

    print(f"loss: {loss}, accuracy: {accuracy}")

    prop_completed, all_frames = evaluate_dt_agent(
        trajectory_data_set, dt, make_env, device=device, max_len=args.max_len, trajectories=args.n_test_episodes)
