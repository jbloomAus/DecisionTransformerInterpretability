import gymnasium as gym
from minigrid.wrappers import OneHotPartialObsWrapper, FullyObsWrapper

def make_env(
    env_id: str, 
    seed: int, 
    idx: int, 
    capture_video: bool, 
    run_name: str, 
    render_mode="rgb_array", 
    max_steps=100, 
    fully_observed=False, 
    flat_one_hot=False,
    video_frequency=50
    ):
    """Return a function that returns an environment after setting up boilerplate."""

    # only one of fully observed or flat one hot can be true.
    assert not (fully_observed and flat_one_hot), "Can't have both fully_observed and flat_one_hot."

    def thunk():

        kwargs = {}
        if render_mode:
            kwargs["render_mode"] = render_mode
        if max_steps:
            kwargs["max_steps"] = max_steps

        env = gym.make(env_id, **kwargs)

        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(
                    env,
                    f"videos/{run_name}",
                    episode_trigger=lambda x: x % video_frequency == 0,  # Video every 50 runs for env #1
                    disable_logger=True
                )

        # hard code for now!
        if env_id.startswith("MiniGrid"):
            if fully_observed:
                env = FullyObsWrapper(env)
            elif flat_one_hot:
                env = OneHotPartialObsWrapper(env)

        obs = env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        env.run_name = run_name
        return env

    return thunk