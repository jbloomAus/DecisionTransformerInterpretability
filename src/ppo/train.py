import torch as t
from tqdm.autonotebook import tqdm
import os
import wandb

from .agent import Agent
from .memory import Memory
from .utils import PPOArgs

device = t.device("cuda" if t.cuda.is_available() else "cpu")


def get_printable_output_for_probe_envs(args: PPOArgs, agent: Agent, probe_idx: int, update: int, num_updates: int):
    """Tests a probe environment and returns a widget-style output for the environment.

    The function prints and returns output showing how well the agent performs on a given probe environment. The output
    includes the agent's computed values and expected values for the observations in the probe environment, as well as
    the agent's computed probabilities and the expected action for the probe environment, if applicable.

    Args:
    - args: a PPOArgs object containing arguments for the PPO training algorithm
    - agent: an Agent object containing a trained PPO agent
    - probe_idx: the index of the probe environment to test
    - update: the current training update number
    - num_updates: the total number of training updates

    Returns:
    - output: a string representing the output widget for the probe environment
    """
    obs_for_probes = [[[0.0]], [[-1.0], [+1.0]],
                      [[0.0], [1.0]], [[0.0]], [[0.0], [1.0]]]
    expected_value_for_probes = [
        1.0, [-1.0, +1.0], [args.gamma, 1.0], 1.0, [1.0, 1.0]]
    expected_actions_for_probs = [None, None, None, 1, [0, 1]]

    obs = t.tensor(obs_for_probes[probe_idx]).to(device)
    output = ""

    # Check if the value is what you expect
    value = agent.critic(obs).detach().cpu().numpy().squeeze()
    expected_value = expected_value_for_probes[probe_idx]
    output += f"Obs: {update+1}/{num_updates}\n\nActual value: {value}\nExpected value: {expected_value}"
    # Check if the action is what you expect
    expected_action = expected_actions_for_probs[probe_idx]
    if expected_action is not None:
        logits = agent.actor(obs)
        probs = logits.softmax(-1).detach().cpu().numpy().squeeze()
        probs = str(probs).replace('\n', '')
        output += f"\n\nActual prob: {probs}\nExpected action: {expected_action}"

    return output


def train_ppo(args: PPOArgs, envs, trajectory_writer=None, probe_idx=None):
    """
    Trains a PPO agent on a given environment.

    Args:
    - args: an instance of PPOArgs containing the hyperparameters for training
    - envs: the environment to train on
    - trajectory_writer: an optional object to write trajectories to a file
    - probe_idx: index of probe environment, if training on probe environment

    Returns:
    None
    """


    memory = Memory(envs, args, device)
    agent = Agent(envs, device=device, hidden_dim=args.hidden_dim)
    num_updates = args.total_timesteps // args.batch_size
    if not args.decay_lr:
        end_lr = args.learning_rate
    else:
        end_lr = 0.0
    optimizer, scheduler = agent.make_optimizer(
        num_updates, initial_lr=args.learning_rate, end_lr=end_lr)

    # out = wg.Output(layout={"padding": "15px"})
    # display(out)
    progress_bar = tqdm(range(num_updates), position=0, leave=True)

    if args.track:
        video_path = os.path.join("videos", args.run_name)
        videos = [i for i in os.listdir(video_path) if i.endswith(".mp4")]
        for video in videos:
            os.remove(os.path.join(video_path, video))
        videos = [i for i in os.listdir(video_path) if i.endswith(".mp4")]

    for update in progress_bar:

        agent.rollout(memory, args, envs, trajectory_writer)
        agent.learn(memory, args, optimizer, scheduler)

        if args.track:
            memory.log()
            videos = check_and_upload_new_video(
                video_path=video_path, videos=videos, step=memory.global_step)

        # Print output (different behaviour for probe envs vs normal envs)
        if probe_idx is None:
            output = memory.get_printable_output()

        else:
            output = get_printable_output_for_probe_envs(
                args, agent, probe_idx, update, num_updates)
        if output:
            # with out:
            #     # print(output)
            #     # out.clear_output(wait=True)
            progress_bar.set_description(output)

        memory.reset()

    if trajectory_writer is not None:
        trajectory_writer.tag_terminated_trajectories()
        trajectory_writer.write(upload_to_wandb=args.track)

    envs.close()


def check_and_upload_new_video(video_path, videos, step=None):
    """
    Checks if new videos have been generated in the video path directory since the last check, and if so,
    uploads them to the current WandB run.

    Args:
    - video_path: The path to the directory where the videos are being saved.
    - videos: A list of the names of the videos that have already been uploaded to WandB.
    - step: The current step in the training loop, used to associate the video with the correct timestep.

    Returns:
    - A list of the names of all the videos currently present in the video path directory.
    """

    current_videos = [i for i in os.listdir(video_path) if i.endswith(".mp4")]
    new_videos = [i for i in current_videos if i not in videos]
    if new_videos:
        for new_video in new_videos:
            path_to_video = os.path.join(video_path, new_video)
            wandb.log({"video": wandb.Video(
                path_to_video,
                fps=4,
                caption=new_video,
                format="mp4",
            )}, step=step)
    return current_videos
