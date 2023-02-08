import torch as t
from einops import rearrange, repeat
from torchtyping import TensorType as TT

def shift_rows(arr):
    """
    Helper function for compute_advantages_vectorized

    Given a 1D array like:
        [1, 2, 3]
    this function will return:
        [[1, 2, 3],
         [0, 1, 2],
         [0, 0, 1]]

    If the array has >1D, it treats the later dimensions as batch dims
    """
    L = arr.shape[0]
    output = t.zeros(L, 2*L, *arr.shape[1:]).to(dtype=arr.dtype)
    output[:, :L] = arr[None, :]
    output = rearrange(output, "t1 t2 ... -> (t1 t2) ...")
    output = output[:L*(2*L-1)]
    output = rearrange(output, "(t1 t2) ... -> t1 t2 ...", t1=L)
    output = output[:, :L]

    return output

def compute_advantages_vectorized(
    next_value: TT["env"],  # noqa: F821
    next_done: TT["env"],  # noqa: F821
    rewards: TT["T", "env"], # noqa: F821
    values: TT["T", "env"],  # noqa: F821
    dones: TT["T", "env"],  # noqa: F821
    device: t.device,
    gamma: float,
    gae_lambda: float
) -> TT["T", "env"]: # noqa: F821
    """
    Basic idea (assuming num_envs=1 in this description, but the case generalises):

        create a matrix of discount factors (gamma*lmda)**l, shape (t, l), suitably shifted
        create a matrix of deltas, shape (t, l), suitably shifted
        mask the deltas after the "done" points
        multiply two matrices and sum over l (second dim)
    """
    T, num_envs = rewards.shape
    next_values = t.concat([values[1:], next_value.unsqueeze(0)])
    next_dones = t.concat([dones[1:], next_done.unsqueeze(0)])
    deltas = rewards + gamma * next_values * (1.0 - next_dones) - values

    deltas_repeated = repeat(deltas, "t2 env -> t1 t2 env", t1=T)
    mask = repeat(next_dones, "t2 env -> t1 t2 env", t1=T).to(device)
    mask_uppertri = repeat(t.triu(t.ones(T, T)), "t1 t2 -> t1 t2 env", env=num_envs).to(device)
    mask = mask * mask_uppertri
    mask = 1 - (mask.cumsum(dim=1) > 0).float()
    mask = t.concat([t.ones(T, 1, num_envs).to(device), mask[:, :-1]], dim=1)
    mask = mask * mask_uppertri
    deltas_masked = mask * deltas_repeated

    discount_factors = (gamma * gae_lambda) ** t.arange(T).to(device)
    discount_factors_repeated = repeat(discount_factors, "t -> t env", env=num_envs)
    discount_factors_shifted = shift_rows(discount_factors_repeated).to(device)

    advantages = (discount_factors_shifted * deltas_masked).sum(dim=1)
    return advantages
