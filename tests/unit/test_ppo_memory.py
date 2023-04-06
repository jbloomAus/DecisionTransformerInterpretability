import pytest
import torch
import gymnasium as gym
from dataclasses import dataclass
import torch
from src.ppo.memory import Memory, Minibatch


@pytest.fixture
def online_config():
    @dataclass
    class DummyOnlineConfig:
        use_trajectory_model: bool = False
        hidden_size: int = 64
        total_timesteps: int = 1000
        learning_rate: float = 0.00025
        decay_lr: bool = False,
        num_envs: int = 10
        num_steps: int = 128
        gamma: float = 0.99
        gae_lambda: float = 0.95
        num_minibatches: int = 10
        update_epochs: int = 4
        clip_coef: float = 0.2
        ent_coef: float = 0.01
        vf_coef: float = 0.5
        max_grad_norm: float = 2
        trajectory_path: str = None
        fully_observed: bool = False
        batch_size: int = 64
        minibatch_size: int = 4
        prob_go_from_end: float = 0.0
        device: torch.device = torch.device("cpu")

    return DummyOnlineConfig()


@pytest.fixture
def memory(online_config):
    envs = gym.vector.SyncVectorEnv(
        [lambda: gym.make('MiniGrid-Dynamic-Obstacles-5x5-v0') for _ in range(4)])
    device = torch.device("cpu")
    return Memory(envs, online_config, device)


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


def test_init(memory):

    torch.testing.assert_allclose(memory.next_done, torch.tensor([0, 0, 0, 0]))
    assert memory.next_value is None
    assert memory.device == torch.device("cpu")
    assert memory.global_step == 0
    assert memory.obs_preprocessor is not None
    assert memory.experiences == []
    assert memory.episode_lengths == []
    assert memory.episode_returns == []
    assert memory.vars_to_log == {}


def test_add_no_ending(online_config):

    num_steps = 10
    args = online_config
    online_config.num_steps = num_steps
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


def test_add_end_of_episode(memory):

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


def test_compute_advantages(memory):

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


def test_get_minibatch_indexes(memory):

    info_final = {"final_info": [{"episode": {"l": 1, "r": 1.0}}]}
    done_final = torch.tensor([0.0, 0.0, 0.0])
    info = {}
    done = torch.tensor([0, 0, 0])
    obs = torch.tensor([1.0, 2.0, 3.0])
    action = torch.tensor([1.0, 2.0, 3.0])
    logprob = torch.tensor([1.0, 2.0, 3.0])
    value = torch.tensor([1.0, 2.0, 3.0])
    reward = torch.tensor([1.0, 2.0, 3.0])

    for i in range(8):
        for j in range(3):
            memory.add(info, obs, done, action, logprob, value, reward)
        memory.add(info_final, obs, done_final, action, logprob, value, reward)

    minibatch_indexes = memory.get_minibatch_indexes(
        batch_size=8*4, minibatch_size=8)
    assert len(minibatch_indexes) == 4  # (8*4)/8
    assert len(minibatch_indexes[0]) == 8  # minibatch_size
    # now assert no number appears twice
    flat_indexes = [item for sublist in minibatch_indexes for item in sublist]
    assert len(flat_indexes) == len(set(flat_indexes))


def test_get_minibatches_standard(memory):

    info_final = {"final_info": [{"episode": {"l": 1, "r": 1.0}}]}
    done_final = torch.tensor([0.0, 0.0, 0.0])
    info = {}
    done = torch.tensor([0.0, 0.0, 0.0])
    obs = torch.tensor([1.0, 2.0, 3.0])
    action = torch.tensor([1.0, 2.0, 3.0])
    logprob = torch.tensor([1.0, 2.0, 3.0])
    value = torch.tensor([1.0, 2.0, 3.0])
    reward = torch.tensor([1.0, 2.0, 3.0])

    for i in range(8):
        for j in range(3):
            memory.add(info, obs, done, action, logprob, value, reward)
        memory.add(info_final, obs, done_final, action, logprob, value, reward)

    memory.next_value = torch.tensor([1.0, 2.0, 3.0])
    memory.next_done = torch.tensor([0, 1, 0])
    memory.args.batch_size = 8*4
    memory.args.minibatch_size = 8
    mbs = memory.get_minibatches()
    assert len(mbs) == 4
    assert isinstance(mbs[0], Minibatch)
    assert len(mbs[0].obs) == 8
    assert hasattr(mbs[0], "obs")
    assert hasattr(mbs[0], "actions")
    assert hasattr(mbs[0], "logprobs")
    assert hasattr(mbs[0], "values")
    assert hasattr(mbs[0], "returns")
    assert hasattr(mbs[0], "advantages")
    assert hasattr(mbs[0], "recurrence_memory")
    assert hasattr(mbs[0], "mask")
    assert mbs[0].mask is None
    assert mbs[0].recurrence_memory is None


def test_get_minibatches_recurrance_memory_and_mask(memory):

    info_final = {"final_info": [{"episode": {"l": 1, "r": 1.0}}]}
    done_final = torch.tensor([0.0, 0.0, 0.0])
    info = {}
    done = torch.tensor([0.0, 0.0, 0.0])
    obs = torch.tensor([1.0, 2.0, 3.0])
    action = torch.tensor([1.0, 2.0, 3.0])
    logprob = torch.tensor([1.0, 2.0, 3.0])
    value = torch.tensor([1.0, 2.0, 3.0])
    reward = torch.tensor([1.0, 2.0, 3.0])
    mask = 1 - torch.tensor([0.0, 0.0, 0.0])
    recurrence_memory = torch.tensor([1.0, 2.0, 3.0])

    for i in range(8):
        for j in range(3):
            memory.add(info, obs, done, action, logprob,
                       value, reward, recurrence_memory, mask)
        memory.add(info_final, obs, done_final, action, logprob,
                   value, reward, recurrence_memory, mask)

    memory.next_value = torch.tensor([1.0, 2.0, 3.0])
    memory.next_done = torch.tensor([0, 1, 0])
    memory.args.batch_size = 8*4
    memory.args.minibatch_size = 8
    mbs = memory.get_minibatches()
    assert len(mbs) == 4
    assert isinstance(mbs[0], Minibatch)
    assert len(mbs[0].obs) == 8
    assert len(mbs) == 4
    assert isinstance(mbs[0], Minibatch)
    assert len(mbs[0].obs) == 8
    assert hasattr(mbs[0], "obs")
    assert hasattr(mbs[0], "actions")
    assert hasattr(mbs[0], "logprobs")
    assert hasattr(mbs[0], "values")
    assert hasattr(mbs[0], "returns")
    assert hasattr(mbs[0], "advantages")
    assert hasattr(mbs[0], "recurrence_memory")
    assert hasattr(mbs[0], "mask")
    assert mbs[0].mask is not None
    assert mbs[0].recurrence_memory is not None


def test_get_minibatch_indexes_recurrence(memory):

    info_final = {"final_info": [{"episode": {"l": 1, "r": 1.0}}]}
    done_final = torch.tensor([0.0, 0.0, 0.0])
    info = {}
    done = {}
    obs = torch.tensor([1.0, 2.0, 3.0])
    action = torch.tensor([1.0, 2.0, 3.0])
    logprob = torch.tensor([1.0, 2.0, 3.0])
    value = torch.tensor([1.0, 2.0, 3.0])
    reward = torch.tensor([1.0, 2.0, 3.0])

    for i in range(8):
        for j in range(3):
            memory.add(info, obs, done, action, logprob, value, reward)
        memory.add(info_final, obs, done_final, action, logprob, value, reward)

    minibatch_indexes = memory.get_minibatch_indexes(
        batch_size=8*4, minibatch_size=8, recurrence=2)
    assert len(minibatch_indexes) == 4  # num_minibatches = (8*4)/(8*2)
    assert len(minibatch_indexes[0]) == 4  # minibatch_size
    # now assert no number appears twice
    flat_indexes = [item for sublist in minibatch_indexes for item in sublist]
    assert len(flat_indexes) == len(set(flat_indexes))
    # no assert that the indexes are seperated by recurrence
    flat_indexes = sorted(flat_indexes)
    for i in range(len(flat_indexes)-1):
        assert flat_indexes[i+1] - flat_indexes[i] == 2


def test_get_minibatches_recurrence(memory):

    info_final = {"final_info": [{"episode": {"l": 1, "r": 1.0}}]}
    done_final = torch.tensor([0.0, 0.0, 0.0])
    info = {}
    done = torch.tensor([0.0, 0.0, 0.0])
    obs = torch.tensor([1.0, 2.0, 3.0])
    action = torch.tensor([1.0, 2.0, 3.0])
    logprob = torch.tensor([1.0, 2.0, 3.0])
    value = torch.tensor([1.0, 2.0, 3.0])
    reward = torch.tensor([1.0, 2.0, 3.0])
    recurrence_memory = torch.tensor([1.0, 2.0, 3.0])
    mask = 1 - done
    mask_final = 1 - done_final

    n = 0
    for i in range(8):
        for j in range(3):
            memory.add(info, obs+n, done, action+n, logprob,
                       value, reward, recurrence_memory, mask)
            n += 1
        memory.add(info_final, obs+n, done_final, action+n, logprob,
                   value, reward, recurrence_memory, mask_final)
        n += 1

    memory.next_value = torch.tensor([1.0, 2.0, 3.0])
    memory.next_done = torch.tensor([0, 1, 0])
    memory.args.batch_size = 8*4
    memory.args.minibatch_size = 8
    mbs = memory.get_minibatches(recurrence=2)
    # minibatch_size // recurrence (there'll be an inner loop that runs recurrence times)
    assert len(mbs) == 4  # hence why this isn't num_minibatches
    assert isinstance(mbs[0], Minibatch)
    assert len(mbs[0].obs) == 4  # minibatch_size

    for i in range(len(mbs)):
        # tensor assert actions match obs
        assert torch.all(mbs[i].actions == mbs[i].obs)


def test_get_minibatches_given_indices_contiguity_of_obs(memory):

    info_final = {"final_info": [{"episode": {"l": 1, "r": 1.0}}]}
    done_final = torch.tensor([1.0, 1.0, 1.0])
    info = {}
    done = torch.tensor([0.0, 0.0, 0.0])
    obs = torch.tensor([0.0, 100.0, -100.0])
    action = torch.tensor([1.0, 2.0, 3.0])
    logprob = torch.tensor([1.0, 2.0, 3.0])
    value = torch.tensor([1.0, 2.0, 3.0])
    reward = torch.tensor([1.0, 2.0, 3.0])
    recurrence_memory = torch.tensor([1.0, 2.0, 3.0])
    mask = 1 - done
    mask_final = 1 - done_final

    n = 0
    for i in range(32):
        for j in range(3):
            print(obs+n)
            memory.add(info, obs+n, done, action+n, logprob,
                       value, reward, recurrence_memory, mask)
            n = 1 + n
        print(obs+n)
        memory.add(info_final, obs+n, done_final, action+n, logprob,
                   value, reward, recurrence_memory, mask_final)
        n = 1 + n

    memory.next_value = torch.tensor([1.0, 2.0, 3.0])
    memory.next_done = torch.tensor([0, 1, 0])
    memory.args.batch_size = 8*4
    memory.args.minibatch_size = 8

    minibatch_indexes = memory.get_minibatch_indexes(
        batch_size=8*4, minibatch_size=8, recurrence=2)
    mbs = memory.get_minibatches(indexes=minibatch_indexes)
    # minibatch_size // recurrence (there'll be an inner loop that runs recurrence times)
    assert len(mbs) == 4  # hence why this isn't num_minibatches
    assert isinstance(mbs[0], Minibatch)
    assert len(mbs[0].obs) == 4  # minibatch_size

    mbs_neg_1 = memory.get_minibatches(indexes=minibatch_indexes-1)
    mbs2 = memory.get_minibatches(indexes=minibatch_indexes+1)

    # assert (1 + mbs[0].obs) == mbs2[0].obs
    for i in range(len(mbs)):
        torch.testing.assert_allclose((1 + mbs[i].obs)*mbs2[i].mask, mbs2[i].obs*mbs2[i].mask,
                                      msg="i:{}\n1+mbs[i].obs: {}\nmbs2[i].obs: {},\nmbs[i].mask: {},\nmbs2[i].mask: {}\n{}".format(
            i,
            1+mbs[i].obs,
            mbs2[i].obs,
            mbs[i].mask,
            mbs2[i].mask,
            minibatch_indexes[i]))


def test_get_minibatches_given_indices_single_env_contiguity_of_obs(memory):

    info_final = {"final_info": [{"episode": {"l": 1, "r": 1.0}}]}
    done_final = torch.tensor([0.0])
    info = {}
    done = torch.tensor([0.0])
    obs = torch.tensor([1.0])
    action = torch.tensor([1.0])
    logprob = torch.tensor([1.0])
    value = torch.tensor([1.0])
    reward = torch.tensor([1.0])
    recurrence_memory = torch.tensor([1.0])
    mask = 1 - done
    mask_final = 1 - done_final

    n = 0
    for i in range(8*3):
        for j in range(3):
            memory.add(info, obs+n, done, action, logprob,
                       value, reward, recurrence_memory, mask)
            n += 1
        memory.add(info_final, obs+n, done_final, action+n, logprob,
                   value, reward, recurrence_memory, mask_final)
        n = 1 + n

    memory.next_value = torch.tensor([1.0])
    memory.next_done = torch.tensor([0])
    memory.args.batch_size = 8*4
    memory.args.minibatch_size = 8

    minibatch_indexes = memory.get_minibatch_indexes(
        batch_size=8*4, minibatch_size=8, recurrence=2)
    mbs = memory.get_minibatches(indexes=minibatch_indexes)
    # minibatch_size // recurrence (there'll be an inner loop that runs recurrence times)
    assert len(mbs) == 4  # hence why this isn't num_minibatches
    assert isinstance(mbs[0], Minibatch)
    assert len(mbs[0].obs) == 4  # minibatch_size

    mbs2 = memory.get_minibatches(indexes=minibatch_indexes+1)

    # assert (1 + mbs[0].obs) == mbs2[0].obs
    for i in range(len(mbs)):
        torch.testing.assert_allclose(1 + mbs[i].obs, mbs2[i].obs)
