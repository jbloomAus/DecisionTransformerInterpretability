import pytest
import gymnasium as gym
import minigrid
import torch
from src.ppo.utils import PPOArgs
from src.ppo.memory import Memory, Minibatch


@pytest.fixture
def memory():
    envs = gym.vector.SyncVectorEnv(
        [lambda: gym.make('MiniGrid-Dynamic-Obstacles-5x5-v0') for _ in range(4)])
    args = PPOArgs()
    device = torch.device("cpu")
    return Memory(envs, args, device)


def test_minibatch_class():

    obs = torch.tensor([1.0, 2.0, 3.0])
    actions = torch.tensor([1.0, 2.0, 3.0])
    logprobs = torch.tensor([1.0, 2.0, 3.0])
    advantages = torch.tensor([1.0, 2.0, 3.0])
    values = torch.tensor([1.0, 2.0, 3.0])
    returns = torch.tensor([1.0, 2.0, 3.0])

    minibatch = Minibatch(
        obs=obs,
        actions=actions,
        logprobs=logprobs,
        advantages=advantages,
        values=values,
        returns=returns
    )

    torch.testing.assert_allclose(minibatch.obs, obs)
    torch.testing.assert_allclose(minibatch.actions, actions)
    torch.testing.assert_allclose(minibatch.logprobs, logprobs)
    torch.testing.assert_allclose(minibatch.advantages, advantages)
    torch.testing.assert_allclose(minibatch.values, values)
    torch.testing.assert_allclose(minibatch.returns, returns)


def test_memory_init(memory):

    torch.testing.assert_allclose(memory.next_done, torch.tensor([0, 0, 0, 0]))
    assert memory.next_value is None
    assert memory.device == torch.device("cpu")
    assert memory.global_step == 0
    assert memory.obs_preprocessor is not None
    assert memory.experiences == []
    assert memory.episode_lengths == []
    assert memory.episode_returns == []
    assert memory.vars_to_log == {}


def test_memory_add_no_ending():

    num_steps = 10
    args = PPOArgs(num_steps=num_steps)
    envs = gym.vector.SyncVectorEnv(
        [lambda: gym.make('CartPole-v0') for _ in range(args.num_envs)])
    memory = Memory(envs=envs, args=args, device="cpu")

    info = {}
    obs = torch.tensor([1.0, 2.0, 3.0])
    done = torch.tensor([1.0, 2.0, 3.0])
    action = torch.tensor([1.0, 2.0, 3.0])
    logprob = torch.tensor([1.0, 2.0, 3.0])
    value = torch.tensor([1.0, 2.0, 3.0])
    reward = torch.tensor([1.0, 2.0, 3.0])

    memory.add(info, obs, done, action, logprob, value, reward)

    # memory doesn't increment global_step if there is no info on an episode ending
    assert memory.global_step == 0
    assert len(memory.experiences) == 1
    assert len(memory.episode_lengths) == 0
    assert len(memory.episode_returns) == 0
    assert memory.vars_to_log == {}


def test_memory_add_end_of_episode(memory):

    info = {"final_info": [{"episode": {"l": 1, "r": 1.0}}]}
    obs = torch.tensor([1.0, 2.0, 3.0])
    done = torch.tensor([1.0, 2.0, 3.0])
    action = torch.tensor([1.0, 2.0, 3.0])
    logprob = torch.tensor([1.0, 2.0, 3.0])
    value = torch.tensor([1.0, 2.0, 3.0])
    reward = torch.tensor([1.0, 2.0, 3.0])

    memory.add(info, obs, done, action, logprob, value, reward)

    # memory doesn't increment global_step if there is no info on an episode ending
    assert memory.global_step == 1
    assert len(memory.experiences) == 1
    assert len(memory.episode_lengths) == 1
    assert len(memory.episode_returns) == 1
    assert memory.vars_to_log == {
        0: {'episode_length': 1, 'episode_return': 1.0}}


def test_memory_compute_advantages(memory):

    info = {"final_info": [{"episode": {"l": 1, "r": 1.0}}]}
    obs = torch.tensor([1.0, 2.0, 3.0])
    done = torch.tensor([1.0, 2.0, 3.0])
    action = torch.tensor([1.0, 2.0, 3.0])
    logprob = torch.tensor([1.0, 2.0, 3.0])
    value = torch.tensor([1.0, 2.0, 3.0])
    reward = torch.tensor([1.0, 2.0, 3.0])

    memory.add(info, obs, done, action, logprob, value, reward)

    advantages = memory.compute_advantages(
        next_value=torch.tensor([1.0, 2.0, 3.0]),
        next_done=torch.tensor([0, 1, 0]),
        rewards=torch.tensor([1.0, 2.0, 3.0]).repeat(2, 1),
        values=torch.tensor([1.0, 2.0, 3.0]).repeat(2, 1),
        dones=torch.tensor([1, 0, 0]).repeat(2, 1),
        device=torch.device("cpu"),
        gamma=0.99,
        gae_lambda=0.95)

    torch.testing.assert_allclose(
        advantages,
        torch.tensor([[0.0, 1.98, 5.7633], [0.99, 0.00, 2.97]])
    )


def test_memory_get_obs_traj(memory):

    info_final = {"final_info": [{"episode": {"l": 1, "r": 1.0}}]}

    info = {}
    obs = torch.tensor([1.0, 2.0, 3.0])
    done = torch.tensor([0.0, 1.0, 0.0, 1.0])
    action = torch.tensor([1.0, 2.0, 3.0])
    logprob = torch.tensor([1.0, 2.0, 3.0])
    value = torch.tensor([1.0, 2.0, 3.0])
    reward = torch.tensor([1.0, 2.0, 3.0])

    for i in range(10):
        memory.add(info, obs + i, done, action, logprob, value, reward)

    memory.add(info_final, obs + 10, done, action, logprob, value, reward)

    obs_traj = memory.get_obs_traj(steps=3, pad_to_length=3)

    # assert shape matches
    assert obs_traj.shape == (3, 3)

    torch.testing.assert_allclose(
        obs_traj,
        torch.tensor([
            [9.0, 10.0, 11.0],
            [10.0, 11.0, 12.0],
            [11.0, 12.0, 13.0]
        ]).T
    )


def test_memory_get_obs_traj_padding_required(memory):

    info_final = {"final_info": [{"episode": {"l": 1, "r": 1.0}}]}

    info = {}
    obs = torch.tensor([1.0, 2.0, 3.0, 4.0])
    done = torch.tensor([0.0, 1.0, 0.0, 1.0])
    action = torch.tensor([1.0, 2.0, 3.0, 4.0])
    logprob = torch.tensor([1.0, 2.0, 3.0, 4.0])
    value = torch.tensor([1.0, 2.0, 3.0, 4.0])
    reward = torch.tensor([1.0, 2.0, 3.0, 4.0])

    for i in range(10):
        memory.add(info, obs, done, action, logprob, value, reward)

    memory.add(info_final, obs, done, action, logprob, value, reward)

    obs_traj = memory.get_obs_traj(steps=3, pad_to_length=10)

    # assert shape matches
    assert obs_traj.shape == (4, 10)

    torch.testing.assert_allclose(
        obs_traj,
        torch.tensor([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0]
        ]).T
    )


def test_memory_get_obs_traj_truncation_required(memory):

    info_final = {"final_info": [{"episode": {"l": 1, "r": 1.0}}]}

    info = {}
    obs = torch.tensor([1.0, 2.0, 3.0, 4.0])
    done = torch.tensor([0.0, 1.0, 0.0, 1.0])
    action = torch.tensor([1.0, 2.0, 3.0, 4.0])
    logprob = torch.tensor([1.0, 2.0, 3.0, 4.0])
    value = torch.tensor([1.0, 2.0, 3.0, 4.0])
    reward = torch.tensor([1.0, 2.0, 3.0, 4.0])

    for i in range(10):
        memory.add(info, obs, done, action, logprob, value, reward)

    memory.add(info_final, obs, done, action, logprob, value, reward)

    obs_traj = memory.get_obs_traj(steps=14, pad_to_length=10)

    # assert shape matches
    assert obs_traj.shape == (4, 10)

    torch.testing.assert_allclose(
        obs_traj,
        torch.tensor([
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0]
        ]).T
    )


def test_memory_get_act_traj(memory):

    info_final = {"final_info": [{"episode": {"l": 1, "r": 1.0}}]}

    info = {}
    obs = torch.tensor([1.0, 2.0, 3.0])
    done = torch.tensor([0.0, 1.0, 0.0, 1.0])
    action = torch.tensor([1.0, 2.0, 3.0])
    logprob = torch.tensor([1.0, 2.0, 3.0])
    value = torch.tensor([1.0, 2.0, 3.0])
    reward = torch.tensor([1.0, 2.0, 3.0])

    for i in range(10):
        memory.add(info, obs, done, action, logprob, value, reward)

    memory.add(info_final, obs, done, action, logprob, value, reward)

    obs_traj = memory.get_act_traj(steps=3, pad_to_length=3)

    # assert shape matches
    assert obs_traj.shape == (3, 3)

    torch.testing.assert_allclose(
        obs_traj,
        torch.tensor([
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0]
        ]).T
    )


def test_memory_get_act_traj_padding_required(memory):

    info_final = {"final_info": [{"episode": {"l": 1, "r": 1.0}}]}

    info = {}
    obs = torch.tensor([1.0, 2.0, 3.0, 4.0])
    done = torch.tensor([0.0, 1.0, 0.0, 1.0])
    action = torch.tensor([1.0, 2.0, 3.0, 4.0])
    logprob = torch.tensor([1.0, 2.0, 3.0, 4.0])
    value = torch.tensor([1.0, 2.0, 3.0, 4.0])
    reward = torch.tensor([1.0, 2.0, 3.0, 4.0])

    for i in range(10):
        memory.add(info, obs, done, action, logprob, value, reward)

    memory.add(info_final, obs, done, action, logprob, value, reward)

    obs_traj = memory.get_act_traj(steps=3, pad_to_length=10)

    # assert shape matches
    assert obs_traj.shape == (4, 10)

    torch.testing.assert_allclose(
        obs_traj,
        torch.tensor([
            [3.0, 3.0, 3.0, 3.0],
            [3.0, 3.0, 3.0, 3.0],
            [3.0, 3.0, 3.0, 3.0],
            [3.0, 3.0, 3.0, 3.0],
            [3.0, 3.0, 3.0, 3.0],
            [3.0, 3.0, 3.0, 3.0],
            [3.0, 3.0, 3.0, 3.0],
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0]
        ]).T
    )


def test_memory_get_act_traj_truncation_required(memory):

    info_final = {"final_info": [{"episode": {"l": 1, "r": 1.0}}]}

    info = {}
    obs = torch.tensor([1.0, 2.0, 3.0, 4.0])
    done = torch.tensor([0.0, 1.0, 0.0, 1.0])
    action = torch.tensor([1.0, 2.0, 3.0, 4.0])
    logprob = torch.tensor([1.0, 2.0, 3.0, 4.0])
    value = torch.tensor([1.0, 2.0, 3.0, 4.0])
    reward = torch.tensor([1.0, 2.0, 3.0, 4.0])

    for i in range(10):
        memory.add(info, obs, done, action, logprob, value, reward)

    memory.add(info_final, obs, done, action, logprob, value, reward)

    obs_traj = memory.get_act_traj(steps=14, pad_to_length=10)

    # assert shape matches
    assert obs_traj.shape == (4, 10)

    torch.testing.assert_allclose(
        obs_traj,
        torch.tensor([
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0]
        ]).T
    )
