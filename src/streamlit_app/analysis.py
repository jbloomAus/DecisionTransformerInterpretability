import streamlit as st
import torch as t


def name_residual_components(dt, cache):
    """
    Returns a list of keys for the residual components of the decision transformer which contribute to the final residual decomp
    """
    result = [
        "input_tokens",  # this will be block.0.resid_pre - hook_pos_embed
        "hook_pos_embed",
    ]

    n_layers = dt.transformer_config.n_layers

    # start with input
    for layer in range(n_layers):
        # add the residual components from the attention layer
        result.append(f"blocks.{layer}.attn.hook_z")
        result.append(f"transformer.blocks.{layer}.attn.b_O")
        # add the residual components from the mlp layer
        result.append(f"blocks.{layer}.hook_mlp_out")

    return result


def get_residual_decomp(
    dt,
    cache,
    logit_dir,
    nice_names=True,
    seq_pos=-1,
    include_attention_bias=False,
):
    """
    Returns the residual decomposition for the decision transformer
    """
    decomp = {}
    n_heads = dt.transformer_config.n_heads
    state_dict = dt.state_dict()
    # get the residual components
    residual_components = name_residual_components(dt, cache)

    for component in residual_components:
        if component == "hook_pos_embed":
            decomp[component] = cache[component][:, seq_pos] @ logit_dir
        elif component == "input_tokens":
            decomp[component] = (
                (cache["blocks.0.hook_resid_pre"] - cache["hook_pos_embed"])
                @ logit_dir
            )[:, seq_pos]
        elif component.endswith(".hook_z"):
            for head in range(n_heads):
                layer = int(component.split(".")[1])
                output = (
                    cache[component][:, seq_pos, head, :]
                    @ dt.transformer.blocks[layer].attn.W_O[head]
                )
                decomp[component + f".{head}"] = output @ logit_dir

        elif component.endswith(".hook_mlp_out"):
            decomp[component] = cache[component][:, seq_pos, :] @ logit_dir
        elif component.endswith(".b_O"):
            if include_attention_bias:
                decomp[component] = state_dict[component] @ logit_dir

    for k in decomp:
        decomp[k] = decomp[k].detach().numpy()

    if nice_names:
        decomp = get_nice_names(decomp)

    return decomp


def get_nice_names(decomp):
    """
    Will update each dictionary key with a nicer string and remove the old key
    """
    new_decomp = {}
    for k in decomp.keys():
        if k == "hook_pos_embed":
            new_decomp["Positional Embedding"] = decomp[k]
        elif k == "input_tokens":
            new_decomp["Input Tokens"] = decomp[k]
        elif ".hook_z" in k:
            layer = int(k.split(".")[1])
            head = int(k.split(".")[-1])
            new_decomp[f"Attention Layer {layer} Head {head}"] = decomp[k]
        elif k.endswith(".hook_mlp_out"):
            layer = int(k.split(".")[1])
            new_decomp[f"MLP Layer {layer}"] = decomp[k]
        elif k.endswith(".b_O"):
            layer = int(k.split(".")[2])
            new_decomp[f"Attention Bias Layer {layer}"] = decomp[k]

    return new_decomp
