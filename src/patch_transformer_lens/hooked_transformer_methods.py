"""
Some of the methods in TransformerLens make assumptions which I've invalidated.
Ideally, we should refactor the code to remove these assumptions, but for now
I'm just going to monkey patch the methods to make them work.
"""

import torch
import einops
from typing import Dict


# Layer norm assumes the existence of an unembed which I have converted to nn.Identity()
# I will patch it to check this! rather than just ignore it.
# this is used in load decision transforrmer
def fold_layer_norm(self, state_dict: Dict[str, torch.Tensor]):
    """Takes in a state dict from a pretrained model, formatted to be consistent with HookedTransformer but with LayerNorm weights and biases. Folds these into the neighbouring weights. See further_comments.md for more details

    Args:
        state_dict (Dict[str, torch.Tensor]): State dict of pretrained model
    """
    for l in range(self.cfg.n_layers):
        # Fold ln1 into attention - it's important to fold biases first,
        # since biases depend on weights but not vice versa
        # The various indexing is just to broadcast ln.b and ln.w along every axis other than d_model. Each weight matrix right multiplies.
        # To fold in the bias, we use the W_ matrix to map it to the hidden space of the layer, so we need to sum along axis -2, which is the residual stream space axis.
        state_dict[f"blocks.{l}.attn.b_Q"] = state_dict[
            f"blocks.{l}.attn.b_Q"
        ] + (
            state_dict[f"blocks.{l}.attn.W_Q"]
            * state_dict[f"blocks.{l}.ln1.b"][None, :, None]
        ).sum(
            -2
        )
        state_dict[f"blocks.{l}.attn.b_K"] = state_dict[
            f"blocks.{l}.attn.b_K"
        ] + (
            state_dict[f"blocks.{l}.attn.W_K"]
            * state_dict[f"blocks.{l}.ln1.b"][None, :, None]
        ).sum(
            -2
        )
        state_dict[f"blocks.{l}.attn.b_V"] = state_dict[
            f"blocks.{l}.attn.b_V"
        ] + (
            state_dict[f"blocks.{l}.attn.W_V"]
            * state_dict[f"blocks.{l}.ln1.b"][None, :, None]
        ).sum(
            -2
        )

        state_dict[f"blocks.{l}.attn.W_Q"] = (
            state_dict[f"blocks.{l}.attn.W_Q"]
            * state_dict[f"blocks.{l}.ln1.w"][None, :, None]
        )
        state_dict[f"blocks.{l}.attn.W_K"] = (
            state_dict[f"blocks.{l}.attn.W_K"]
            * state_dict[f"blocks.{l}.ln1.w"][None, :, None]
        )
        state_dict[f"blocks.{l}.attn.W_V"] = (
            state_dict[f"blocks.{l}.attn.W_V"]
            * state_dict[f"blocks.{l}.ln1.w"][None, :, None]
        )

        # Finally, we center the weights reading from the residual stream. The output of the first
        # part of the LayerNorm is mean 0 and standard deviation 1, so the mean of any input vector
        # of the matrix doesn't matter and can be set to zero.
        # Equivalently, the output of LayerNormPre is orthogonal to the vector of all 1s (because
        # dotting with that gets the sum), so we can remove the component of the matrix parallel to this.
        state_dict[f"blocks.{l}.attn.W_Q"] -= einops.reduce(
            state_dict[f"blocks.{l}.attn.W_Q"],
            "head_index d_model d_head -> head_index 1 d_head",
            "mean",
        )
        state_dict[f"blocks.{l}.attn.W_K"] -= einops.reduce(
            state_dict[f"blocks.{l}.attn.W_K"],
            "head_index d_model d_head -> head_index 1 d_head",
            "mean",
        )
        state_dict[f"blocks.{l}.attn.W_V"] -= einops.reduce(
            state_dict[f"blocks.{l}.attn.W_V"],
            "head_index d_model d_head -> head_index 1 d_head",
            "mean",
        )

        del (
            state_dict[f"blocks.{l}.ln1.w"],
            state_dict[f"blocks.{l}.ln1.b"],
        )

        # Fold ln2 into MLP
        if not self.cfg.attn_only:
            state_dict[f"blocks.{l}.mlp.b_in"] = state_dict[
                f"blocks.{l}.mlp.b_in"
            ] + (
                state_dict[f"blocks.{l}.mlp.W_in"]
                * state_dict[f"blocks.{l}.ln2.b"][:, None]
            ).sum(
                -2
            )
            state_dict[f"blocks.{l}.mlp.W_in"] = (
                state_dict[f"blocks.{l}.mlp.W_in"]
                * state_dict[f"blocks.{l}.ln2.w"][:, None]
            )

            # Center the weights that read in from the LayerNormPre
            state_dict[f"blocks.{l}.mlp.W_in"] -= einops.reduce(
                state_dict[f"blocks.{l}.mlp.W_in"],
                "d_model d_mlp -> 1 d_mlp",
                "mean",
            )

            del (
                state_dict[f"blocks.{l}.ln2.w"],
                state_dict[f"blocks.{l}.ln2.b"],
            )

            if self.cfg.act_fn.startswith("solu"):
                # Fold ln3 into activation
                state_dict[f"blocks.{l}.mlp.b_out"] = state_dict[
                    f"blocks.{l}.mlp.b_out"
                ] + (
                    state_dict[f"blocks.{l}.mlp.W_out"]
                    * state_dict[f"blocks.{l}.mlp.ln.b"][:, None]
                ).sum(
                    -2
                )
                state_dict[f"blocks.{l}.mlp.W_out"] = (
                    state_dict[f"blocks.{l}.mlp.W_out"]
                    * state_dict[f"blocks.{l}.mlp.ln.w"][:, None]
                )

                # Center the weights that read in from the LayerNormPre
                state_dict[f"blocks.{l}.mlp.W_out"] -= einops.reduce(
                    state_dict[f"blocks.{l}.mlp.W_out"],
                    "d_mlp d_model -> 1 d_model",
                    "mean",
                )
                del (
                    state_dict[f"blocks.{l}.mlp.ln.w"],
                    state_dict[f"blocks.{l}.mlp.ln.b"],
                )
    # Fold ln_final into Unembed

    if not self.cfg.final_rms:
        # Dumb bug from my old SoLU training code, some models have RMSNorm instead of LayerNorm pre unembed.

        if "unembed.b_U" in state_dict:
            state_dict[f"unembed.b_U"] = state_dict[f"unembed.b_U"] + (
                state_dict[f"unembed.W_U"] * state_dict[f"ln_final.b"][:, None]
            ).sum(dim=-2)
        else:
            print("WARNING: no unembed.b_U in state_dict, deleting ln_final.b")
        del state_dict[f"ln_final.b"]

    if "unembed.W_U" in state_dict:
        state_dict[f"unembed.W_U"] = (
            state_dict[f"unembed.W_U"] * state_dict[f"ln_final.w"][:, None]
        )

        # Center the weights that read in from the LayerNormPre
        state_dict[f"unembed.W_U"] -= einops.reduce(
            state_dict[f"unembed.W_U"], "d_model d_vocab -> 1 d_vocab", "mean"
        )
    else:
        print(
            "WARNING: no unembed.b_U in state_dict, deleting ln_final.w. You should be folding these weight into subsequent layers before calling this func."
        )

    del state_dict[f"ln_final.w"]
    return state_dict
