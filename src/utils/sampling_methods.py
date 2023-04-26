"""
Provides a series of utilities which take a 
pytorch.distribution.categorical.Categorical
object and return a sampled value according to
to some rule. 

These include:
- Basic (sample according to probs)
- Greedy
- Temperature Sampling
- topK Sampling
- bottomK Sampling
"""
from torch.distributions.categorical import Categorical


def greedy_sample(probs: Categorical):
    """
    Returns the index of the maximum probability
    """
    return probs.probs.argmax(dim=-1)


def basic_sample(probs: Categorical):
    """
    Returns a sample from the distribution
    """
    return probs.sample()


def temp_sample(probs: Categorical, temperature: float):
    """
    Returns a sample from the distribution
    """

    adjusted_logits = probs.logits / temperature

    # Create a new Categorical distribution with the adjusted logits
    adjusted_categorical = Categorical(logits=adjusted_logits)

    # Sample from the new distribution
    return adjusted_categorical.sample()


def topk_sample(probs: Categorical, k: int):
    """
    Returns a sample from the distribution
    """
    top_k_probs, top_k_indices = probs.probs.topk(k, dim=-1)
    top_k_distribution = Categorical(probs=top_k_probs)
    index_in_top_k = top_k_distribution.sample()
    return top_k_indices.gather(-1, index_in_top_k.unsqueeze(-1)).squeeze(-1)


def bottomk_sample(probs: Categorical, k: int):
    """
    Returns a sample from the distribution
    """
    bottom_k_probs, bottom_k_indices = probs.probs.topk(
        k, dim=-1, largest=False
    )
    bottom_k_distribution = Categorical(probs=bottom_k_probs)
    index_in_bottom_k = bottom_k_distribution.sample()
    return bottom_k_indices.gather(
        -1, index_in_bottom_k.unsqueeze(-1)
    ).squeeze(-1)


def sample_from_categorical(probs: Categorical, method: str, **kwargs):
    """
    Returns a sample from the distribution according to the selected method.

    Warning: Don't use anything other than basic when training PPO. This is
    intended to assist demonstration collection and is not a part of the
    PPO algorithm.
    """
    if method == "basic":
        return basic_sample(probs)
    elif method == "greedy":
        return greedy_sample(probs)
    elif method == "temperature":
        return temp_sample(probs, temperature=kwargs.get("temperature"))
    elif method == "topk":
        return topk_sample(probs, k=kwargs.get("k"))
    elif method == "bottomk":
        return bottomk_sample(probs, k=kwargs.get("k"))
    else:
        raise ValueError("Invalid sampling method provided: {}".format(method))
