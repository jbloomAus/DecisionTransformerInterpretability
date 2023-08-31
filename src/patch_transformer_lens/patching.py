"""
A module for patching activations in a transformer model, and measuring the effect of the patch on the output.
This implements the activation patching technique for a range of types of activation. 
The structure is to have a single generic_activation_patch function that does everything, and to have a range of specialised functions for specific types of activation.

See this explanation for more https://dynalist.io/d/n2ZWtnoYHrU1s4vnFSAQ519J#z=qeWBvs-R-taFfcCq-S_hgMqx
And check out the Activation Patching in TransformerLens Demo notebook for a demo of how to use this module.

Notes for Joseph's Monkey patch version of this module:
1. Updated so I can pass in tokenized text instead of token ids. 
2. Remove the return_index_df of generic_activation_patch, since I don't like it. 
3. Use run_with_cache instead of run with hooks, to enable neuron activation based metrics. 
"""

from __future__ import annotations
import torch
from typing import (
    Optional,
    Union,
    Dict,
    Callable,
    Sequence,
    Optional,
    Tuple,
    List,
)
from typing_extensions import Literal
import numpy as np

from transformer_lens import HookedTransformer, ActivationCache
from transformer_lens.hook_points import HookPoint
import transformer_lens.utils as utils
import pandas as pd
import itertools
from functools import partial
from tqdm.auto import tqdm
from jaxtyping import Float, Int

import einops


Logits = torch.Tensor
AxisNames = Literal[
    "layer", "pos", "head_index", "head", "src_pos", "dest_pos"
]


from typing import Sequence

import streamlit as st


def make_df_from_ranges(
    column_max_ranges: Sequence[int], column_names: Sequence[str]
) -> pd.DataFrame:
    """
    Takes in a list of column names and max ranges for each column, and returns a dataframe with the cartesian product of the range for each column (ie iterating through all combinations from zero to column_max_range - 1, in order, incrementing the final column first)
    """
    rows = list(
        itertools.product(
            *[range(axis_max_range) for axis_max_range in column_max_ranges]
        )
    )
    df = pd.DataFrame(rows, columns=column_names)
    return df


CorruptedActivation = torch.Tensor
PatchedActivation = torch.Tensor


def generic_activation_patch(
    model: HookedTransformer,
    corrupted_tokens: Int[torch.Tensor, "batch pos"],  # noqa: F722
    clean_cache: ActivationCache,
    patching_metric: Callable[
        [Float[torch.Tensor, "batch pos d_vocab"]],  # noqa: F722
        Float[torch.Tensor, ""],  # noqa: F722
    ],
    patch_setter: Callable[
        [CorruptedActivation, Sequence[int], ActivationCache],
        PatchedActivation,
    ],
    activation_name: str,
    index_axis_names: Optional[Sequence[AxisNames]] = None,
    index_df: Optional[pd.DataFrame] = None,
    return_cache: bool = False,
    apply_metric_to_cache: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, pd.DataFrame]]:
    """
    A generic function to do activation patching, will be specialised to specific use cases.

    Activation patching is about studying the counterfactual effect of a specific activation between a clean run and a corrupted run. The idea is have two inputs, clean and corrupted, which have two different outputs, and differ in some key detail. Eg "The Eiffel Tower is in" vs "The Colosseum is in". Then to take a cached set of activations from the "clean" run, and a set of corrupted.

    Internally, the key function comes from three things: A list of tuples of indices (eg (layer, position, head_index)), a index_to_act_name function which identifies the right activation for each index, a patch_setter function which takes the corrupted activation, the index and the clean cache, and a metric for how well the patched model has recovered.

    The indices can either be given explicitly as a pandas dataframe, or by listing the relevant axis names and having them inferred from the tokens and the model config. It is assumed that the first column is always layer.

    This function then iterates over every tuple of indices, does the relevant patch, and stores it

    Args:
        model: The relevant model
        corrupted_tokens: The input tokens for the corrupted run
        clean_cache: The cached activations from the clean run
        patching_metric: A function from the model's output logits to some metric (eg loss, logit diff, etc)
        patch_setter: A function which acts on (corrupted_activation, index, clean_cache) to edit the activation and patch in the relevant chunk of the clean activation
        activation_name: The name of the activation being patched
        index_axis_names: The names of the axes to (fully) iterate over, implicitly fills in index_df
        index_df: A dataframe of indices to iterate over, implicitly fills in index_axis_names
        return_cache: Whether to return the patched cache. Defaults to False
        apply_metric_to_cache: Whether to apply the metric to the cache. Defaults to False

    Returns:
        patched_output: The tensor of the patching metric for each patch. By default it has one dimension for each index dimension, via index_df set explicitly it is flattened with one element per row.
        index_df *optional*: The dataframe of indices
    """

    if index_df is None:
        assert index_axis_names is not None

        # Get the max range for all possible axes
        max_axis_range = {
            "layer": model.cfg.n_layers,
            "pos": corrupted_tokens.shape[-1]
            if len(corrupted_tokens.shape) == 2
            else corrupted_tokens.shape[-2],
            "head_index": model.cfg.n_heads,
        }

        max_axis_range["src_pos"] = max_axis_range["pos"]
        max_axis_range["dest_pos"] = max_axis_range["pos"]
        max_axis_range["head"] = max_axis_range["head_index"]

        # Get the max range for each axis we iterate over
        index_axis_max_range = [
            max_axis_range[axis_name] for axis_name in index_axis_names
        ]

        # Get the dataframe where each row is a tuple of indices
        index_df = make_df_from_ranges(index_axis_max_range, index_axis_names)

        flattened_output = False
    else:
        # A dataframe of indices was provided. Verify that we did not *also* receive index_axis_names
        assert index_axis_names is None
        index_axis_max_range = index_df.max().to_list()

        flattened_output = True

    # Create an empty tensor to show the patched metric for each patch
    if flattened_output:
        patched_metric_output = torch.zeros(
            len(index_df), device=model.cfg.device
        )
    else:
        patched_metric_output = torch.zeros(
            index_axis_max_range, device=model.cfg.device
        )

    # A generic patching hook - for each index, it applies the patch_setter appropriately to patch the activation
    def patching_hook(corrupted_activation, hook, index, clean_activation):
        return patch_setter(corrupted_activation, index, clean_activation)

    # Iterate over every list of indices, and make the appropriate patch!
    for c, index_row in enumerate(tqdm((list(index_df.iterrows())))):
        index = index_row[1].to_list()

        # The current activation name is just the activation name plus the layer (assumed to be the first element of the input)
        current_activation_name = utils.get_act_name(
            activation_name, layer=index[0]
        )

        # The hook function cannot receive additional inputs, so we use partial to include the specific index and the corresponding clean activation
        current_hook = partial(
            patching_hook,
            index=index,
            clean_activation=clean_cache[current_activation_name],
        )

        # Run the model with the patching hook and get the logits!
        # with
        # patched_logits = model.run_with_hooks(
        #     corrupted_tokens,
        #     fwd_hooks=[(current_activation_name, current_hook)],
        # )

        with model.hooks(fwd_hooks=[(current_activation_name, current_hook)]):
            patched_logits, patched_cache = model.run_with_cache(
                corrupted_tokens
            )

        # Calculate the patching metric and store
        metric_input = (
            patched_cache if apply_metric_to_cache else patched_logits
        )

        if flattened_output:
            patched_metric_output[c] = patching_metric(metric_input).item()
        else:
            patched_metric_output[tuple(index)] = patching_metric(
                metric_input
            ).item()

    # cache[f"blocks.{layer}.mlp.hook_pre"]
    # st.write("patched_cache", patched_cache.k
    if return_cache:
        return patched_metric_output, patched_cache

    return patched_metric_output


# Defining patch setters for various shapes of activations
def layer_pos_patch_setter(corrupted_activation, index, clean_activation):
    """
    Applies the activation patch where index = [layer, pos]

    Implicitly assumes that the activation axis order is [batch, pos, ...], which is true of everything that is not an attention pattern shaped tensor.
    """
    assert len(index) == 2
    layer, pos = index
    corrupted_activation[:, pos, ...] = clean_activation[:, pos, ...]
    return corrupted_activation


def layer_pos_head_vector_patch_setter(
    corrupted_activation,
    index,
    clean_activation,
):
    """
    Applies the activation patch where index = [layer, pos, head_index]

    Implicitly assumes that the activation axis order is [batch, pos, head_index, ...], which is true of all attention head vector activations (q, k, v, z, result) but *not* of attention patterns.
    """
    assert len(index) == 3
    layer, pos, head_index = index
    corrupted_activation[:, pos, head_index] = clean_activation[
        :, pos, head_index
    ]
    return corrupted_activation


def layer_head_vector_patch_setter(
    corrupted_activation,
    index,
    clean_activation,
):
    """
    Applies the activation patch where index = [layer,  head_index]

    Implicitly assumes that the activation axis order is [batch, pos, head_index, ...], which is true of all attention head vector activations (q, k, v, z, result) but *not* of attention patterns.
    """
    assert len(index) == 2
    layer, head_index = index
    corrupted_activation[:, :, head_index] = clean_activation[:, :, head_index]

    return corrupted_activation


def layer_head_pattern_patch_setter(
    corrupted_activation,
    index,
    clean_activation,
):
    """
    Applies the activation patch where index = [layer,  head_index]

    Implicitly assumes that the activation axis order is [batch, head_index, dest_pos, src_pos], which is true of attention scores and patterns.
    """
    assert len(index) == 2
    layer, head_index = index
    corrupted_activation[:, head_index, :, :] = clean_activation[
        :, head_index, :, :
    ]

    return corrupted_activation


def layer_head_pos_pattern_patch_setter(
    corrupted_activation,
    index,
    clean_activation,
):
    """
    Applies the activation patch where index = [layer,  head_index, dest_pos]

    Implicitly assumes that the activation axis order is [batch, head_index, dest_pos, src_pos], which is true of attention scores and patterns.
    """
    assert len(index) == 3
    layer, head_index, dest_pos = index
    corrupted_activation[:, head_index, dest_pos, :] = clean_activation[
        :, head_index, dest_pos, :
    ]

    return corrupted_activation


def layer_head_dest_src_pos_pattern_patch_setter(
    corrupted_activation,
    index,
    clean_activation,
):
    """
    Applies the activation patch where index = [layer,  head_index, dest_pos, src_pos]

    Implicitly assumes that the activation axis order is [batch, head_index, dest_pos, src_pos], which is true of attention scores and patterns.
    """
    assert len(index) == 4
    layer, head_index, dest_pos, src_pos = index
    corrupted_activation[:, head_index, dest_pos, src_pos] = clean_activation[
        :, head_index, dest_pos, src_pos
    ]

    return corrupted_activation


# Defining activation patching functions for a range of common activation patches.
get_act_patch_resid_pre = partial(
    generic_activation_patch,
    patch_setter=layer_pos_patch_setter,
    activation_name="resid_pre",
    index_axis_names=("layer", "pos"),
)
get_act_patch_resid_pre.__doc__ = """
    Function to get activation patching results for the residual stream (at the start of each block) (by position). Returns a tensor of shape [n_layers, pos]

    See generic_activation_patch for a more detailed explanation of activation patching 

    Args:
        model: The relevant model
        corrupted_tokens (torch.Tensor): The input tokens for the corrupted run. Has shape [batch, pos]
        clean_cache (ActivationCache): The cached activations from the clean run
        patching_metric: A function from the model's output logits to some metric (eg loss, logit diff, etc)

    Returns:
        patched_output (torch.Tensor): The tensor of the patching metric for each resid_pre patch. Has shape [n_layers, pos]
    """

get_act_patch_resid_mid = partial(
    generic_activation_patch,
    patch_setter=layer_pos_patch_setter,
    activation_name="resid_mid",
    index_axis_names=("layer", "pos"),
)
get_act_patch_resid_mid.__doc__ = """
    Function to get activation patching results for the residual stream (between the attn and MLP layer of each block) (by position). Returns a tensor of shape [n_layers, pos]

    See generic_activation_patch for a more detailed explanation of activation patching 

    Args:
        model: The relevant model
        corrupted_tokens (torch.Tensor): The input tokens for the corrupted run. Has shape [batch, pos]
        clean_cache (ActivationCache): The cached activations from the clean run
        patching_metric: A function from the model's output logits to some metric (eg loss, logit diff, etc)

    Returns:
        patched_output (torch.Tensor): The tensor of the patching metric for each patch. Has shape [n_layers, pos]
    """

get_act_patch_attn_out = partial(
    generic_activation_patch,
    patch_setter=layer_pos_patch_setter,
    activation_name="attn_out",
    index_axis_names=("layer", "pos"),
)
get_act_patch_attn_out.__doc__ = """
    Function to get activation patching results for the output of each Attention layer (by position). Returns a tensor of shape [n_layers, pos]

    See generic_activation_patch for a more detailed explanation of activation patching 

    Args:
        model: The relevant model
        corrupted_tokens (torch.Tensor): The input tokens for the corrupted run. Has shape [batch, pos]
        clean_cache (ActivationCache): The cached activations from the clean run
        patching_metric: A function from the model's output logits to some metric (eg loss, logit diff, etc)

    Returns:
        patched_output (torch.Tensor): The tensor of the patching metric for each patch. Has shape [n_layers, pos]
    """

get_act_patch_mlp_out = partial(
    generic_activation_patch,
    patch_setter=layer_pos_patch_setter,
    activation_name="mlp_out",
    index_axis_names=("layer", "pos"),
)
get_act_patch_mlp_out.__doc__ = """
    Function to get activation patching results for the output of each MLP layer (by position). Returns a tensor of shape [n_layers, pos]

    See generic_activation_patch for a more detailed explanation of activation patching 

    Args:
        model: The relevant model
        corrupted_tokens (torch.Tensor): The input tokens for the corrupted run. Has shape [batch, pos]
        clean_cache (ActivationCache): The cached activations from the clean run
        patching_metric: A function from the model's output logits to some metric (eg loss, logit diff, etc)

    Returns:
        patched_output (torch.Tensor): The tensor of the patching metric for each patch. Has shape [n_layers, pos]
    """

get_act_patch_attn_head_out_by_pos = partial(
    generic_activation_patch,
    patch_setter=layer_pos_head_vector_patch_setter,
    activation_name="z",
    index_axis_names=("layer", "pos", "head"),
)
get_act_patch_attn_head_out_by_pos.__doc__ = """
    Function to get activation patching results for the output of each Attention Head (by position). Returns a tensor of shape [n_layers, pos, n_heads]

    See generic_activation_patch for a more detailed explanation of activation patching 

    Args:
        model: The relevant model
        corrupted_tokens (torch.Tensor): The input tokens for the corrupted run. Has shape [batch, pos]
        clean_cache (ActivationCache): The cached activations from the clean run
        patching_metric: A function from the model's output logits to some metric (eg loss, logit diff, etc)

    Returns:
        patched_output (torch.Tensor): The tensor of the patching metric for each patch. Has shape [n_layers, pos, n_heads]
    """

get_act_patch_attn_head_q_by_pos = partial(
    generic_activation_patch,
    patch_setter=layer_pos_head_vector_patch_setter,
    activation_name="q",
    index_axis_names=("layer", "pos", "head"),
)
get_act_patch_attn_head_q_by_pos.__doc__ = """
    Function to get activation patching results for the queries of each Attention Head (by position). Returns a tensor of shape [n_layers, pos, n_heads]

    See generic_activation_patch for a more detailed explanation of activation patching 

    Args:
        model: The relevant model
        corrupted_tokens (torch.Tensor): The input tokens for the corrupted run. Has shape [batch, pos]
        clean_cache (ActivationCache): The cached activations from the clean run
        patching_metric: A function from the model's output logits to some metric (eg loss, logit diff, etc)

    Returns:
        patched_output (torch.Tensor): The tensor of the patching metric for each patch. Has shape [n_layers, pos, n_heads]
    """

get_act_patch_attn_head_k_by_pos = partial(
    generic_activation_patch,
    patch_setter=layer_pos_head_vector_patch_setter,
    activation_name="k",
    index_axis_names=("layer", "pos", "head"),
)
get_act_patch_attn_head_k_by_pos.__doc__ = """
    Function to get activation patching results for the keys of each Attention Head (by position). Returns a tensor of shape [n_layers, pos, n_heads]

    See generic_activation_patch for a more detailed explanation of activation patching 

    Args:
        model: The relevant model
        corrupted_tokens (torch.Tensor): The input tokens for the corrupted run. Has shape [batch, pos]
        clean_cache (ActivationCache): The cached activations from the clean run
        patching_metric: A function from the model's output logits to some metric (eg loss, logit diff, etc)

    Returns:
        patched_output (torch.Tensor): The tensor of the patching metric for each patch. Has shape [n_layers, pos, n_heads]
    """

get_act_patch_attn_head_v_by_pos = partial(
    generic_activation_patch,
    patch_setter=layer_pos_head_vector_patch_setter,
    activation_name="v",
    index_axis_names=("layer", "pos", "head"),
)
get_act_patch_attn_head_v_by_pos.__doc__ = """
    Function to get activation patching results for the values of each Attention Head (by position). Returns a tensor of shape [n_layers, pos, n_heads]

    See generic_activation_patch for a more detailed explanation of activation patching 

    Args:
        model: The relevant model
        corrupted_tokens (torch.Tensor): The input tokens for the corrupted run. Has shape [batch, pos]
        clean_cache (ActivationCache): The cached activations from the clean run
        patching_metric: A function from the model's output logits to some metric (eg loss, logit diff, etc)

    Returns:
        patched_output (torch.Tensor): The tensor of the patching metric for each patch. Has shape [n_layers, pos, n_heads]
    """

get_act_patch_attn_head_pattern_by_pos = partial(
    generic_activation_patch,
    patch_setter=layer_head_pos_pattern_patch_setter,
    activation_name="pattern",
    index_axis_names=("layer", "head_index", "dest_pos"),
)
get_act_patch_attn_head_pattern_by_pos.__doc__ = """
    Function to get activation patching results for the attention pattern of each Attention Head (by destination position). Returns a tensor of shape [n_layers, n_heads, dest_pos]

    See generic_activation_patch for a more detailed explanation of activation patching 

    Args:
        model: The relevant model
        corrupted_tokens (torch.Tensor): The input tokens for the corrupted run. Has shape [batch, pos]
        clean_cache (ActivationCache): The cached activations from the clean run
        patching_metric: A function from the model's output logits to some metric (eg loss, logit diff, etc)

    Returns:
        patched_output (torch.Tensor): The tensor of the patching metric for each patch. Has shape [n_layers, n_heads, dest_pos]
    """

get_act_patch_attn_head_pattern_dest_src_pos = partial(
    generic_activation_patch,
    patch_setter=layer_head_dest_src_pos_pattern_patch_setter,
    activation_name="pattern",
    index_axis_names=("layer", "head_index", "dest_pos", "src_pos"),
)
get_act_patch_attn_head_pattern_dest_src_pos.__doc__ = """
    Function to get activation patching results for each destination, source entry of the attention pattern for each Attention Head. Returns a tensor of shape [n_layers, n_heads, dest_pos, src_pos]

    See generic_activation_patch for a more detailed explanation of activation patching 

    Args:
        model: The relevant model
        corrupted_tokens (torch.Tensor): The input tokens for the corrupted run. Has shape [batch, pos]
        clean_cache (ActivationCache): The cached activations from the clean run
        patching_metric: A function from the model's output logits to some metric (eg loss, logit diff, etc)

    Returns:
        patched_output (torch.Tensor): The tensor of the patching metric for each patch. Has shape [n_layers, n_heads, dest_pos, src_pos]
    """


get_act_patch_attn_head_out_all_pos = partial(
    generic_activation_patch,
    patch_setter=layer_head_vector_patch_setter,
    activation_name="z",
    index_axis_names=("layer", "head"),
)
get_act_patch_attn_head_out_all_pos.__doc__ = """
    Function to get activation patching results for the outputs of each Attention Head (across all positions). Returns a tensor of shape [n_layers, n_heads]

    See generic_activation_patch for a more detailed explanation of activation patching 

    Args:
        model: The relevant model
        corrupted_tokens (torch.Tensor): The input tokens for the corrupted run. Has shape [batch, pos]
        clean_cache (ActivationCache): The cached activations from the clean run
        patching_metric: A function from the model's output logits to some metric (eg loss, logit diff, etc)

    Returns:
        patched_output (torch.Tensor): The tensor of the patching metric for each patch. Has shape [n_layers, n_heads]
    """

get_act_patch_attn_head_q_all_pos = partial(
    generic_activation_patch,
    patch_setter=layer_head_vector_patch_setter,
    activation_name="q",
    index_axis_names=("layer", "head"),
)
get_act_patch_attn_head_q_all_pos.__doc__ = """
    Function to get activation patching results for the queries of each Attention Head (across all positions). Returns a tensor of shape [n_layers, n_heads]

    See generic_activation_patch for a more detailed explanation of activation patching 

    Args:
        model: The relevant model
        corrupted_tokens (torch.Tensor): The input tokens for the corrupted run. Has shape [batch, pos]
        clean_cache (ActivationCache): The cached activations from the clean run
        patching_metric: A function from the model's output logits to some metric (eg loss, logit diff, etc)

    Returns:
        patched_output (torch.Tensor): The tensor of the patching metric for each patch. Has shape [n_layers, n_heads]
    """

get_act_patch_attn_head_k_all_pos = partial(
    generic_activation_patch,
    patch_setter=layer_head_vector_patch_setter,
    activation_name="k",
    index_axis_names=("layer", "head"),
)
get_act_patch_attn_head_k_all_pos.__doc__ = """
    Function to get activation patching results for the keys of each Attention Head (across all positions). Returns a tensor of shape [n_layers, n_heads]

    See generic_activation_patch for a more detailed explanation of activation patching 

    Args:
        model: The relevant model
        corrupted_tokens (torch.Tensor): The input tokens for the corrupted run. Has shape [batch, pos]
        clean_cache (ActivationCache): The cached activations from the clean run
        patching_metric: A function from the model's output logits to some metric (eg loss, logit diff, etc)

    Returns:
        patched_output (torch.Tensor): The tensor of the patching metric for each patch. Has shape [n_layers, n_heads]
    """

get_act_patch_attn_head_v_all_pos = partial(
    generic_activation_patch,
    patch_setter=layer_head_vector_patch_setter,
    activation_name="v",
    index_axis_names=("layer", "head"),
)
get_act_patch_attn_head_v_all_pos.__doc__ = """
    Function to get activation patching results for the values of each Attention Head (across all positions). Returns a tensor of shape [n_layers, n_heads]

    See generic_activation_patch for a more detailed explanation of activation patching 

    Args:
        model: The relevant model
        corrupted_tokens (torch.Tensor): The input tokens for the corrupted run. Has shape [batch, pos]
        clean_cache (ActivationCache): The cached activations from the clean run
        patching_metric: A function from the model's output logits to some metric (eg loss, logit diff, etc)

    Returns:
        patched_output (torch.Tensor): The tensor of the patching metric for each patch. Has shape [n_layers, n_heads]
    """

get_act_patch_attn_head_pattern_all_pos = partial(
    generic_activation_patch,
    patch_setter=layer_head_pattern_patch_setter,
    activation_name="pattern",
    index_axis_names=("layer", "head_index"),
)
get_act_patch_attn_head_pattern_all_pos.__doc__ = """
    Function to get activation patching results for the attention pattern of each Attention Head (across all positions). Returns a tensor of shape [n_layers, n_heads]

    See generic_activation_patch for a more detailed explanation of activation patching 

    Args:
        model: The relevant model
        corrupted_tokens (torch.Tensor): The input tokens for the corrupted run. Has shape [batch, pos]
        clean_cache (ActivationCache): The cached activations from the clean run
        patching_metric: A function from the model's output logits to some metric (eg loss, logit diff, etc)

    Returns:
        patched_output (torch.Tensor): The tensor of the patching metric for each patch. Has shape [n_layers, n_heads]
    """


def get_act_patch_attn_head_all_pos_every(
    model, corrupted_tokens, clean_cache, metric, apply_metric_to_cache=False
) -> Float[torch.Tensor, "patch_type layer head"]:  # noqa: F722
    """Helper function to get activation patching results for every head (across all positions) for every act type (output, query, key, value, pattern). Wrapper around each's patching function, returns a stacked tensor of shape [5, n_layers, n_heads]

    Args:
        model: The relevant model
        corrupted_tokens (torch.Tensor): The input tokens for the corrupted run. Has shape [batch, pos]
        clean_cache (ActivationCache): The cached activations from the clean run
        metric: A function from the model's output logits to some metric (eg loss, logit diff, etc)

    Returns:
        patched_output (torch.Tensor): The tensor of the patching metric for each patch. Has shape [5, n_layers, n_heads]
    """
    act_patch_results = []
    act_patch_results.append(
        get_act_patch_attn_head_out_all_pos(
            model,
            corrupted_tokens,
            clean_cache,
            metric,
            apply_metric_to_cache=apply_metric_to_cache,
        )
    )
    act_patch_results.append(
        get_act_patch_attn_head_q_all_pos(
            model,
            corrupted_tokens,
            clean_cache,
            metric,
            apply_metric_to_cache=apply_metric_to_cache,
        )
    )
    act_patch_results.append(
        get_act_patch_attn_head_k_all_pos(
            model,
            corrupted_tokens,
            clean_cache,
            metric,
            apply_metric_to_cache=apply_metric_to_cache,
        )
    )
    act_patch_results.append(
        get_act_patch_attn_head_v_all_pos(
            model,
            corrupted_tokens,
            clean_cache,
            metric,
            apply_metric_to_cache=apply_metric_to_cache,
        )
    )
    act_patch_results.append(
        get_act_patch_attn_head_pattern_all_pos(
            model,
            corrupted_tokens,
            clean_cache,
            metric,
            apply_metric_to_cache=apply_metric_to_cache,
        )
    )
    return torch.stack(act_patch_results, dim=0)


def get_act_patch_attn_head_by_pos_every(
    model, corrupted_tokens, clean_cache, metric, apply_metric_to_cache=False
) -> Float[torch.Tensor, "patch_type layer pos head"]:  # noqa: F722
    """Helper function to get activation patching results for every head (by position) for every act type (output, query, key, value, pattern). Wrapper around each's patching function, returns a stacked tensor of shape [5, n_layers, pos, n_heads]

    Args:
        model: The relevant model
        corrupted_tokens (torch.Tensor): The input tokens for the corrupted run. Has shape [batch, pos]
        clean_cache (ActivationCache): The cached activations from the clean run
        metric: A function from the model's output logits to some metric (eg loss, logit diff, etc)

    Returns:
        patched_output (torch.Tensor): The tensor of the patching metric for each patch. Has shape [5, n_layers, pos, n_heads]
    """
    act_patch_results = []
    act_patch_results.append(
        get_act_patch_attn_head_out_by_pos(
            model,
            corrupted_tokens,
            clean_cache,
            metric,
            apply_metric_to_cache=apply_metric_to_cache,
        )
    )
    act_patch_results.append(
        get_act_patch_attn_head_q_by_pos(
            model,
            corrupted_tokens,
            clean_cache,
            metric,
            apply_metric_to_cache=apply_metric_to_cache,
        )
    )
    act_patch_results.append(
        get_act_patch_attn_head_k_by_pos(
            model,
            corrupted_tokens,
            clean_cache,
            metric,
            apply_metric_to_cache=apply_metric_to_cache,
        )
    )
    act_patch_results.append(
        get_act_patch_attn_head_v_by_pos(
            model,
            corrupted_tokens,
            clean_cache,
            metric,
            apply_metric_to_cache=apply_metric_to_cache,
        )
    )

    # Reshape pattern to be compatible with the rest of the results
    pattern_results = get_act_patch_attn_head_pattern_by_pos(
        model,
        corrupted_tokens,
        clean_cache,
        metric,
        apply_metric_to_cache=apply_metric_to_cache,
    )
    act_patch_results.append(
        einops.rearrange(pattern_results, "batch head pos -> batch pos head")
    )
    return torch.stack(act_patch_results, dim=0)


def get_act_patch_block_every(
    model, corrupted_tokens, clean_cache, metric, apply_metric_to_cache=False
) -> Float[torch.Tensor, "patch_type layer pos"]:  # noqa: F722
    """Helper function to get activation patching results for the residual stream (at the start of each block), output of each Attention layer and output of each MLP layer. Wrapper around each's patching function, returns a stacked tensor of shape [3, n_layers, pos]

    Args:
        model: The relevant model
        corrupted_tokens (torch.Tensor): The input tokens for the corrupted run. Has shape [batch, pos]
        clean_cache (ActivationCache): The cached activations from the clean run
        metric: A function from the model's output logits to some metric (eg loss, logit diff, etc)

    Returns:
        patched_output (torch.Tensor): The tensor of the patching metric for each patch. Has shape [3, n_layers, pos]
    """
    act_patch_results = []
    act_patch_results.append(
        get_act_patch_resid_pre(
            model,
            corrupted_tokens,
            clean_cache,
            metric,
            apply_metric_to_cache=apply_metric_to_cache,
        )
    )
    act_patch_results.append(
        get_act_patch_attn_out(
            model,
            corrupted_tokens,
            clean_cache,
            metric,
            apply_metric_to_cache=apply_metric_to_cache,
        )
    )
    act_patch_results.append(
        get_act_patch_mlp_out(
            model,
            corrupted_tokens,
            clean_cache,
            metric,
            apply_metric_to_cache=apply_metric_to_cache,
        )
    )
    return torch.stack(act_patch_results, dim=0)


# Path Patching work by Callum McDougal.
# Sent to me for evaluation/Feedback.
# I'm going to test it out in app which will make it easy for me to
# see what it's practically like to use / if the interface is good enough.


def hook_fn_patch_generic(
    clean_activation: Float[torch.Tensor, "batch seq_pos ..."],  # noqa: F722
    hook: HookPoint,
    patching_cache: ActivationCache,
    seq_pos: Union[int, List[int]] = None,
) -> Float[torch.Tensor, "batch seq_pos ..."]:  # noqa: F722
    """
    Function which patches the entire activation (possibly at a subset of sequence positions).

    This is useful in step 3 of path patching, if our receiver components are not specific to a head.
    It's also useful in step 2 of path patching, because we might want to patch the entire activation.
    """
    if seq_pos is None:
        seq_pos = list(range(clean_activation.shape[1]))
    clean_activation[:, seq_pos] = patching_cache[hook.name][:, seq_pos]
    return clean_activation


def hook_fn_patch_head_vector(
    clean_activation: Float[
        torch.Tensor, "batch pos head_index d_head"  # noqa: F722
    ],
    hook: HookPoint,
    patching_cache: ActivationCache,
    heads_to_patch: List[Union[int, Tuple[int, int]]],
    seq_pos: Union[int, List[int]] = None,
) -> Float[torch.Tensor, "batch pos head_index d_head"]:  # noqa: F722
    """
    Function which patches the activation vector at specific heads (possibly also at a subset of sequence positions).

    If `heads_to_patch` is list of ints, these are interpreted as heads in the current layer
    If `heads_to_patch` is list of tuples, these are interpreted as (layer, head) pairs

    This is useful in step 3 of path patching, if our receiver components are specific to a head.
    """
    batch = list(range(clean_activation.shape[0]))
    if seq_pos is None:
        seq_pos = list(range(clean_activation.shape[1]))
    heads_to_patch = (
        heads_to_patch
        if isinstance(heads_to_patch[0], int)
        else [head for layer, head in heads_to_patch if layer == hook.layer()]
    )
    idx = np.ix_(batch, seq_pos, heads_to_patch)
    clean_activation[idx] = patching_cache[hook.name][idx]
    return clean_activation


# def hook_fn_patch_or_freeze_head_vector(
#     head_vector: Float[torch.Tensor, "batch pos head_index d_head"],
#     hook: HookPoint,
#     patching_cache: ActivationCache,
#     freezing_cache: ActivationCache,
#     heads_to_patch: List[Tuple[int, int]],
#     seq_pos: Union[int, List[int]] = None,
# ) -> Float[torch.Tensor, "batch pos head_index d_head"]:
#     '''
#     head_vector:
#         this can be q, k or v

#     heads_to_patch: list of (layer, head) tuples
#         we patch (i.e. set to patching cache value) all heads in `heads_to_patch`
#         we freeze (i.e. set to freezing cache value) all heads not in this list

#     This is useful in step 2 of path patching, because we'll have to patch some heads and freeze others.
#     '''
#     # Setting using ..., otherwise changing head_vector will edit cache value too
#     if seq_pos is None: seq_pos = list(range(clean_activation.shape[1]))
#     heads_to_patch_in_this_layer = [head for layer, head in heads_to_patch if layer == hook.layer()]
#     head_vector[:] = freezing_cache[hook.name][:]
#     head_vector[:, seq_pos, heads_to_patch_in_this_layer] = patching_cache[hook.name][:, seq_pos, heads_to_patch_in_this_layer]
#     return head_vector


def path_patch(
    model: HookedTransformer,
    corrupted_tokens: Int[torch.Tensor, "batch pos"],  # noqa: F722
    clean_tokens: Int[torch.Tensor, "batch pos"],  # noqa: F722
    patching_metric: Callable,
    corrupted_cache: Optional[ActivationCache] = None,
    clean_cache: Optional[ActivationCache] = None,
    sender_components: Union[str, List[str]] = "z",
    sender_seq_pos: Literal["each", "all"] = "each",
    receiver_components: List[Tuple] = [],
    receiver_seq_pos: Union[Literal["all"], int, List[int]] = "all",
    verbose: bool = False,
):
    """
    This function supports a variety of path patching methods.

    Path patching from (sender components -> receiver components) involves the following algorithm:
        (1) Gather all activations on clean and corrupted distributions.
        (2) Run model on clean with sender components patched from corrupted and all other components frozen. Cache the receiver components.
        (3) Run model on clean with receiver components patched from previously cached values.

    Result of this algorithm: the model behaves like it does on the clean distribution, but with every direct path from sender -> receiver
    patched (where we define a direct path as one which doesn't go through any other attention heads, only MLPs or skip connections).


    Arguments:

        receiver_components: List[Tuple]
            This contains all the receiver components. Accepted syntax for receiver components is:
                (layer: int, head: int)                  -> patch all inputs to that head
                (layer: int, head: int, input_type: str) -> patch a specific input to that head (input_type can be "q", "k", "v", "pattern")
                (layer, activation_name: str)            -> patch at a specific activation (e.g. could be "resid_pre", "attn_out", "mlp_out")

        receiver_seq_pos: Union[Literal["all"], int]
            "all"          -> then we patch the receiver at all sequence positions
            int, List[int] -> patch the receiver at only those sequence positions

        sender_components: Union[str, List[str]]
            Special one:
                "all_blocks" -> result has shape (3, layer), because we're patching at every block (resid_pre, attn_out, mlp_out) in every layer
            If neither of these:
                Then this is assumed to be a component name (e.g. "resid_pre" or "attn_out" or "z") or list of component names
                And we iterate through all these component names when we patch senders
                Result will have shape (n_components, layer)
                Note that if "z" is included in this list, this means "by individual head", so this adds 12 to n_components (if you want to patch all
                heads at once then use "attn_out")

        sender_seq_pos: Literal["each", "all"]
            If "each", then we patch each position separately (this can be computationally expensive!)
            If "all", then we patch all positions at once


    With the flexibility that all these 4 arguments provide, we can reproduce most of the activation patching methods defined via partial functions above. To give examples,
    if we had receiver_components = [(-1, "resid_post")] and receiver_seq_pos = "all" then this is equiv to activation patching, and we have the following equivalences:

        get_act_patch_resid_pre; returns (layer, seq_pos)
            sender_components = "resid_pre"
            sender_seq_pos = "each"

        get_act_patch_block_every; returns (blocks=3, layer, seq_pos)
            sender_components = "all_blocks"
            sender_seq_pos = "each"

        get_act_patch_attn_head_out_all_pos; returns (layer, head)
            sender_components = "z"
            sender_seq_pos = "all"

    The answer will always be a tensor of between 1 and 4 dimensions.
    Note the hierarchy of dimension orders in the final answer:
        blocks > layer > head > seq_pos


    Things which haven't been added but maybe should be:
        patching on individual neurons in MLPs
    """
    assert (
        len(receiver_components) > 0
    ), "Must specify at least one receiver component"
    assert sender_components in [
        "resid_pre",
        "resid_mid",
        "resid_post",
        "mlp_out",
        "attn_out",
        "z",
        "all_blocks",
    ], f"Invalid sender_components '{sender_components}'"
    batch, seq_len = corrupted_tokens.shape[0], corrupted_tokens.shape[1]

    model.reset_hooks()

    # ========== Step 1 ==========
    # Gather activations on clean and corrupted distributions (we only need attn heads)
    z_name_filter = lambda name: name.endswith("z")
    if corrupted_cache is None:
        # corrupted cache might be used in a hard to predict way so just get the whole thing cause it's easier
        _, corrupted_cache = model.run_with_cache(
            corrupted_tokens,
            # names_filter=lambda name: name.endswith("z"),
            return_type=None,
        )
    if clean_cache is None:
        # clean_cache is only ever used for freezing heads
        _, clean_cache = model.run_with_cache(
            clean_tokens, names_filter=z_name_filter, return_type=None
        )

    # ========== Step 3 preprocessing ==========
    # Process receiver_components, converting them into a form we can use to perform path patching

    # We want a list of hooks for doing patching on the receiver components. This list should have `patching_cache` as a free arg
    # which we'll set later using partial (because this will be the cache we get from step 2).

    # Figure out which sequence positions we're patching
    if receiver_seq_pos == "all":
        receiver_seq_pos = None
    elif isinstance(receiver_seq_pos, int):
        receiver_seq_pos = [receiver_seq_pos]
    # Create list to store all hooks (this is a bit messy because sometimes we patch at individual heads)
    receiver_hook_names_and_fns = []
    receiver_heads_to_patch = {"v": [], "q": [], "k": []}
    for receiver_component in receiver_components:
        # Each component should be (layer, head), or (layer, head, input_type), or (layer, activation_name)
        layer, *component_details = receiver_component
        if layer < 0:
            layer = model.cfg.n_layers + layer
        if isinstance(component_details[0], str):
            # case (layer, activation_name)
            activation_name = component_details[0]
            receiver_hook_names_and_fns.append(
                (
                    utils.get_act_name(activation_name, layer),
                    partial(hook_fn_patch_generic, seq_pos=receiver_seq_pos),
                )
            )
        elif isinstance(component_details[0], int):
            if len(component_details) == 1:
                # case (layer, head)
                head = component_details[0]
                for input_type in "qkv":
                    receiver_heads_to_patch[input_type].append((layer, head))
            elif len(component_details) == 2:
                # case (layer, head, input_type)
                head, input_type = component_details
                assert input_type in [
                    "q",
                    "k",
                    "v",
                    "pattern",
                ], f"Invalid receiver_component '{receiver_component}'"
                if input_type == "pattern":
                    for input_type in "qk":
                        receiver_heads_to_patch[input_type].append(
                            (layer, head)
                        )
                else:
                    receiver_heads_to_patch[input_type].append((layer, head))
            else:
                raise ValueError(
                    f"Invalid receiver_component '{receiver_component}'"
                )
        else:
            raise ValueError(
                f"Invalid receiver_component '{receiver_component}'"
            )
    # This is where we deal with the patching for head inputs, based on the `receiver_heads_to_patch` dict we just built
    for input_type, heads_list in receiver_heads_to_patch.items():
        layers_containing_receiver_heads = list(
            set([layer for layer, head in heads_list])
        )
        for layer in layers_containing_receiver_heads:
            receiver_hook_names_and_fns.append(
                (
                    utils.get_act_name(input_type, layer),
                    partial(
                        hook_fn_patch_head_vector,
                        heads_to_patch=heads_list,
                        seq_pos=receiver_seq_pos,
                    ),
                )
            )
    # Lastly, get all the hook names in a list (so we can create a names filter when we cache during step 2)
    receiver_hook_names = [
        hook_name for hook_name, hook_fn in receiver_hook_names_and_fns
    ]
    receiver_hook_names_filter = lambda name: name in receiver_hook_names

    # ========== Step 2 preprocessing ==========
    # Process sender_components, converting them into a form we can use to perform path patching

    # We want a list of "head freezing hooks" and "component patching hooks". During step 2, for each possible sender component in the
    # latter list, we'll be patching this sender component, and freezing all heads which aren't sender components. It's also useful at
    # this stage to figure out what the shape of our final output will be, and the names of the dimensions.

    # Figure out which sequence positions we're patching
    if sender_seq_pos == "all":
        sender_seq_pos_list = [[i for i in range(seq_len)]]
    elif sender_seq_pos == "each":
        sender_seq_pos_list = [[i] for i in range(seq_len)]
    else:
        raise ValueError(f"Invalid sender_seq_pos value '{sender_seq_pos}'")
    # Dictionary to store "head freezing hooks" and "component patching hooks"
    sender_hooks = {
        "freezing": [
            (
                z_name_filter,
                partial(hook_fn_patch_generic, patching_cache=clean_cache),
            )
        ],
        "patching": [],
    }
    # Convert our sender components into a list of activation names
    if isinstance(sender_components, str):
        sender_components = [sender_components]
    sender_components_list = []
    for sender_component in sender_components:
        if sender_component == "all_blocks":
            sender_components_list.extend(["resid_pre", "attn_out", "mlp_out"])
        elif sender_component == "z":
            sender_components_list.extend(
                [f"L.{head}" for head in range(model.cfg.n_heads)]
            )
        else:
            sender_components_list.append(sender_component)
    # Iterate through this list, and append to sender_hooks["patching"] for each one
    # (we need to deal with "z" separately, because this indicates we're patching by head rather than just one component per layer)
    for sender_component in sender_components_list:
        if "." in sender_component:
            activation_name = "z"
            head = int(sender_component.split(".")[-1])
            hook_fn = partial(
                hook_fn_patch_head_vector,
                patching_cache=corrupted_cache,
                heads_to_patch=[head],
            )
        else:
            activation_name = sender_component
            hook_fn = partial(
                hook_fn_patch_generic, patching_cache=corrupted_cache
            )
        for layer, seq_pos in itertools.product(
            range(model.cfg.n_layers), sender_seq_pos_list
        ):
            sender_hooks["patching"].append(
                (
                    utils.get_act_name(activation_name, layer),
                    partial(hook_fn, seq_pos=seq_pos),
                )
            )
    # Define a list to store results, and the shape we'll eventually get it into (also get dimension descriptions for printing)
    results = []
    results_shape = [model.cfg.n_layers]
    results_shape_desc = ["layers"]
    if len(sender_components_list) > 1:
        results_shape.insert(0, len(sender_components_list))
        results_shape_desc.insert(0, "/".join(sender_components))
    if sender_seq_pos == "each":
        results_shape.append(seq_len)
        results_shape_desc.append("sequence positions")

    # This loop is where we do the actual patching algorithm (steps 2 and 3)
    for i, sender_hook_patching in tqdm(
        list(enumerate(sender_hooks["patching"]))
    ):
        # ========== Step 2 ==========
        # Run on clean distribution, with sender component patched from corrupted, every non-sender head frozen

        for hook_name, hook_fn in sender_hooks["freezing"] + [
            sender_hook_patching
        ]:
            model.add_hook(hook_name, hook_fn)  # , level=1)

        _, temp_cache = model.run_with_cache(
            clean_tokens,
            names_filter=receiver_hook_names_filter,
            return_type=None,
        )
        model.reset_hooks()
        assert set(temp_cache.keys()) == set(receiver_hook_names)

        # ========== Step 3 ==========
        # Run on clean distribution, patching in the receiver components from the results of step 2

        temp_receiver_hook_names_and_fns = [
            (hook_name, partial(hook_fn, patching_cache=temp_cache))
            for (hook_name, hook_fn) in receiver_hook_names_and_fns
        ]
        # print(temp_receiver_hook_names_and_fns)
        patched_logits = model.run_with_hooks(
            clean_tokens,
            fwd_hooks=temp_receiver_hook_names_and_fns,
            return_type="logits",
        )

        # Store the results
        results.append(patching_metric(patched_logits).item())
        model.reset_hooks()

    # Get the results as a tensor, and reshape it appropriately (also get dimension descriptions for printing)
    results = (
        torch.tensor(results, dtype=torch.float32)
        .reshape(results_shape)
        .squeeze()
    )

    # Shape of results?
    # If 3D, we want (component, layer, seq_pos) or (layer, head, seq_pos)
    # If 2D, we want (layer, head) or (component, layer)
    # If 1D, we want (layer)
    # So we just need to swap the first 2 dimensions if we have (head, layer) and actually want (layer, head)
    if (results.ndim >= 2) and (sender_components == ["z"]):
        results = results.transpose(0, 1)
        results_shape_desc[0], results_shape_desc[1] = (
            results_shape_desc[1],
            results_shape_desc[0],
        )

    if verbose:
        print(f"Shape of results: {results.shape}")
        print(f"Dimensions are: {results_shape_desc}")

    return results
