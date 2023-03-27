import torch as t
from torch.distributions.categorical import Categorical
from torchtyping import TensorType as TT
from torchtyping import patch_typeguard
from typeguard import typechecked

patch_typeguard()


def calc_clipped_surrogate_objective(
    probs: Categorical, mb_action: t.Tensor, mb_advantages: t.Tensor, mb_logprobs: t.Tensor, clip_coef: float
) -> t.Tensor:
    '''
    Return the clipped surrogate objective, suitable for maximisation with gradient ascent.

    Args:
        probs (Categorical): A distribution containing the actor's
            unnormalized logits of shape (minibatch, num_actions).
        mb_action (Tensor): A tensor of shape (minibatch,) containing the actions taken by the agent in the minibatch.
        mb_advantages (Tensor): A tensor of shape (minibatch,) containing the
            advantages estimated for each state in the minibatch.
        mb_logprobs (Tensor): A tensor of shape (minibatch,) containing the
            log probabilities of the actions taken by the agent in the minibatch.
        clip_coef (float): Amount of clipping, denoted by epsilon in Eq 7.

    Returns:
        Tensor: The clipped surrogate objective computed over the minibatch, with shape ().

    '''
    logits_diff = probs.log_prob(mb_action) - mb_logprobs

    r_theta = t.exp(logits_diff)

    mb_advantages = (mb_advantages - mb_advantages.mean()) / \
        (mb_advantages.std() + 10e-8)

    non_clipped = r_theta * mb_advantages
    clipped = t.clip(r_theta, 1 - clip_coef, 1 + clip_coef) * mb_advantages

    return t.minimum(non_clipped, clipped).mean()


@typechecked
def calc_value_function_loss(values: TT["batch"], mb_returns: TT["batch"], vf_coef: float) -> t.Tensor:  # noqa: F821
    '''
    Compute the value function portion of the loss function.

    Args:
        values (Tensor): A tensor of shape (minibatch,) containing the value function
            estimates for the states in the minibatch.
        mb_returns (Tensor): A tensor of shape (minibatch,) containing the discounted
            returns estimated for each state in the minibatch.
        vf_coef (float): The coefficient for the value loss, which weights its
            contribution to the overall loss. Denoted by c_1 in the paper.

    Returns:
        Tensor: The value function loss computed over the minibatch, with shape ().

    '''
    return 0.5 * vf_coef * (values - mb_returns).pow(2).mean()


def calc_entropy_bonus(probs: Categorical, ent_coef: float):
    '''
    Return the entropy bonus term, suitable for gradient ascent.

    Args:
        probs (Categorical): A distribution containing the actor's unnormalized
            logits of shape (minibatch, num_actions).
        ent_coef (float): The coefficient for the entropy loss, which weights its
            contribution to the overall loss. Denoted by c_2 in the paper.

    Returns:
        Tensor: The entropy bonus computed over the minibatch, with shape ().
    '''
    return ent_coef * probs.entropy().mean()
