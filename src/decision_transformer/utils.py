import argparse
import json
import math
import types
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from src.config import EnvironmentConfig, TransformerModelConfig
from src.models.trajectory_transformer import (
    CloneTransformer,
    DecisionTransformer,
)
from src.patch_transformer_lens.hooked_transformer_methods import (
    fold_layer_norm,
)


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Decision Transformer",
        description="Train a decision transformer on a trajectory dataset.",
        epilog="The last enemy that shall be defeated is death.",
    )
    parser.add_argument("--exp_name", type=str, default="Dev")
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--trajectory_path", type=str)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--d_mlp", type=int, default=256)
    parser.add_argument("--activation_fn", type=str, default="relu")
    parser.add_argument("--gated_mlp", action=argparse.BooleanOptionalAction)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--n_ctx", type=int, default=3)
    parser.add_argument("--layer_norm", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--train_epochs", type=int, default=10)
    parser.add_argument("--test_epochs", type=int, default=3)
    parser.add_argument("--optimizer", type=str, default="AdamW")
    parser.add_argument(
        "--scheduler", type=str, default="CosineAnnealingWarmup"
    )
    parser.add_argument("--warm_up_steps", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--lr_end", type=float, default=10e-8)
    parser.add_argument("--num_cycles", type=int, default=3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--state_embedding", type=str, default="grid")
    parser.add_argument(
        "--linear_time_embedding",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument("--pct_traj", type=float, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--track",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--wandb_project_name",
        type=str,
        default="DecisionTransformerInterpretability",
    )
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--test_frequency", type=int, default=100)
    parser.add_argument("--eval_frequency", type=int, default=100)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--eval_num_envs", type=int, default=8)
    parser.add_argument(
        "--initial_rtg",
        action="append",
        help="<Required> Set flag",
        required=False,
        default=[],
    )
    parser.add_argument("--prob_go_from_end", type=float, default=0.1)
    parser.add_argument("--eval_max_time_steps", type=int, default=1000)
    parser.add_argument("--cuda", action=argparse.BooleanOptionalAction)
    parser.add_argument(
        "--model_type", type=str, default="decision_transformer"
    )
    parser.add_argument(
        "--convert_to_one_hot",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
    )
    args = parser.parse_args()
    return args


# TODO Support loading Clone Transformers
def load_decision_transformer(
    model_path, env=None, tlens_weight_processing=False
) -> DecisionTransformer:
    """ """

    model_info = torch.load(model_path)
    state_dict = model_info["model_state_dict"]
    transformer_config = TransformerModelConfig(
        **json.loads(model_info["model_config"])
    )

    environment_config = EnvironmentConfig(
        **json.loads(model_info["environment_config"])
    )

    model = DecisionTransformer(
        environment_config=environment_config,
        transformer_config=transformer_config,
    )

    model.load_state_dict(state_dict)

    # we'll need these later
    if model.transformer_config.layer_norm == "LN":
        ln_final_b = state_dict["transformer.ln_final.b"]
        ln_final_w = state_dict["transformer.ln_final.w"]

    if tlens_weight_processing:
        # get the model.transformer state_dict
        state_dict = model.transformer.state_dict()

        # # monkey patch!
        model.transformer.fold_layer_norm = types.MethodType(
            fold_layer_norm, model.transformer
        )

        # do some folding of anything ln_final feeds into before delete ln_final in patched ln_fold code.

        # use the patched method to fold the layer norm and do other stuff.

        # the state dict must come from teh origina model with ln
        # but the recieving model needs to have LN_Pre.

        if model.model_type == "decision_transformer":
            if model.transformer_config.layer_norm == "LN":
                model.transformer_config.layer_norm = "LNPre"
                model.transformer = (
                    model.initialize_easy_transformer()
                )  # reinitallize it to not use LN
                # monkey patch
                model.transformer.fold_layer_norm = types.MethodType(
                    fold_layer_norm, model.transformer
                )

        # Not currently supporting Clone Transformers in app.
        # elif model.model_type == "clone_transformer":
        #     model.transformer = CloneTransformer(environment_config, transformer_config)

        # load and process state dict, expects a new instantiation with LNPre
        model.transformer.load_and_process_state_dict(
            state_dict=state_dict,
            fold_ln=True,
            center_writing_weights=False,
            fold_value_biases=True,
            center_unembed=False,
        )

        # this should work but I'm not done unless I also fold ln_final into the last layers / anything
        # that uses the model output.
        if model.transformer_config.layer_norm == "LNPre":
            new_action_predictor_weight = (
                model.action_predictor.weight * ln_final_w[:, None].T
            )
            new_action_predictor_bias = model.action_predictor.bias + (
                (model.action_predictor.weight * ln_final_b[:, None].T).sum(
                    dim=-1
                )
            )

            new_state_predictor_weight = (
                model.state_predictor.weight * ln_final_w[:, None].T
            )
            new_state_predictor_bias = model.state_predictor.bias + (
                (model.state_predictor.weight * ln_final_b[:, None].T).sum(
                    dim=-1
                )
            )

            new_reward_predictor_weight = (
                model.reward_predictor.weight * ln_final_w[:, None].T
            )
            new_reward_predictor_bias = model.reward_predictor.bias + (
                (model.reward_predictor.weight * ln_final_b[:, None].T).sum(
                    dim=-1
                )
            )

            # load these in
            model.action_predictor.load_state_dict(
                {
                    "weight": new_action_predictor_weight,
                    "bias": new_action_predictor_bias,
                }
            )

            model.state_predictor.load_state_dict(
                {
                    "weight": new_state_predictor_weight,
                    "bias": new_state_predictor_bias,
                }
            )

            model.reward_predictor.load_state_dict(
                {
                    "weight": new_reward_predictor_weight,
                    "bias": new_reward_predictor_bias,
                }
            )

            print(
                "Ignore T-Lens warnings about folding layer norm into unembed, we just did it manually"
            )

    return model


def get_max_len_from_model_type(model_type: str, n_ctx: int):
    """
    Ihe max len in timesteps is 3 for decision transformers
    and 2 for clone transformers since decision transformers
    have 3 tokens per timestep and clone transformers have 2.

    This is a map between timestep and tokens. We start with one
    for the most recent state/action and then add another
    timestep for every 3 tokens for decision transformers and
    every 2 tokens for clone transformers.
    """
    assert model_type in ["decision_transformer", "clone_transformer"]
    if model_type == "decision_transformer":
        return 1 + n_ctx // 3
    else:
        return 1 + n_ctx // 2


def initialize_padding_inputs(
    max_len: int,
    initial_obs: dict,
    initial_rtg: float,
    action_pad_token: int,
    batch_size=1,
    device="cpu",
):
    """
    Initializes input tensors for a decision transformer based on the given maximum length of the sequence, initial observation, initial return-to-go (rtg) value,
    and padding token for actions.

    Padding token for rtg is assumed to be the initial RTG at all values. This is important.
    Padding token for initial obs is 0. But it could be -1 and we might parameterize in the future.
    Mask is initialized to 0.0 and then set to 1.0 for all values that are not padding (one value currently)

    Args:
    - max_len (int): maximum length of the sequence
    - initial_obs (Dict[str, Union[torch.Tensor, np.ndarray]]): initial observation dictionary, containing an "image" tensor with shape (batch_size, channels, height, width)
    - initial_rtg (float): initial return-to-go value used to initialize the reward-to-go tensor
    - action_pad_token (int): padding token used to initialize the actions tensor
    - batch_size (int): batch size of the sequences (default: 1)

    Returns:
    - obs (torch.Tensor): tensor of shape (batch_size, max_len, channels, height, width), initialized with zeros and the initial observation in the last dimension
    - actions (torch.Tensor): tensor of shape (batch_size, max_len - 1, 1), initialized with the padding token
    - reward (torch.Tensor): tensor of shape (batch_size, max_len, 1), initialized with zeros
    - rtg (torch.Tensor): tensor of shape (1, max_len, 1), initialized with the initial rtg value and broadcasted to the batch size dimension
    - timesteps (torch.Tensor): tensor of shape (batch_size, max_len, 1), initialized with zeros
    - mask (torch.Tensor): tensor of shape (batch_size, max_len), initialized with zeros and ones at the last position to mark the end of the sequence
    """

    device = torch.device(device)

    mask = torch.concat(
        (
            torch.zeros((batch_size, max_len - 1), dtype=torch.bool),
            torch.ones((batch_size, 1), dtype=torch.bool),
        ),
        dim=1,
    ).to(device)

    obs_dim = initial_obs["image"].shape[-3:]
    if len(initial_obs["image"].shape) == 3:
        assert (
            batch_size == 1
        ), "only one initial obs provided but batch size > 1"
        obs_image = torch.tensor(initial_obs["image"])[None, None, :, :, :].to(
            device
        )
    elif len(initial_obs["image"].shape) == 4:
        obs_image = torch.tensor(initial_obs["image"])[:, None, :, :, :].to(
            device
        )
    else:
        raise ValueError(
            "initial obs image has invalid shape: {}".format(
                initial_obs["image"].shape
            )
        )

    obs = torch.concat(
        (
            torch.zeros((batch_size, max_len - 1, *obs_dim)).to(device),
            obs_image,
        ),
        dim=1,
    ).to(float)

    reward = torch.zeros((batch_size, max_len, 1), dtype=torch.float).to(
        device
    )
    rtg = initial_rtg * torch.ones(
        (batch_size, max_len, 1), dtype=torch.float
    ).to(device)
    timesteps = torch.zeros((batch_size, max_len, 1), dtype=torch.long).to(
        device
    )

    actions = (
        torch.ones(batch_size, max_len - 1, 1, dtype=torch.long)
        * action_pad_token
    ).to(device)

    return obs, actions, reward, rtg, timesteps, mask


# Use MinGPT code for optimizer configuration
def configure_optimizers(model, offline_config):
    """
    https://github.com/karpathy/minGPT/blob/37baab71b9abea1b76ab957409a1cc2fbfba8a26/mingpt/model.py#LL215C1-L258C25

    This long function is unfortunately doing something very simple and is being very defensive:
    We are separating out all parameters of the model into two buckets: those that will experience
    weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
    We are then returning the PyTorch optimizer object.
    """

    optim_groups = get_optim_groups(model, offline_config)

    optimizer = get_optimizer(
        offline_config.optimizer,
        optim_groups,
        lr=offline_config.lr,
        weight_decay=offline_config.weight_decay,
    )
    return optimizer


def get_optim_groups(model, offline_config):
    # separate out all parameters to those that will and won't experience regularizing weight decay
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (
        torch.nn.Linear,
    )  # not going to work trivially for HookedTransformer
    blacklist_weight_modules = (
        torch.nn.LayerNorm,
        torch.nn.Embedding,
    )  # not going to work trivially for HookedTransformer
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name
            # random note: because named_modules and named_parameters are recursive
            # we will see the same tensors p many many times. but doing it this way
            # allows us to know which parent module any tensor p belongs to...
            if pn.endswith("bias") or "b_" in pn:
                # all biases will not be decayed
                no_decay.add(fpn)
            elif "embedding" in pn:
                # all embedding weights will not be decayed
                # state_embedding and reward embedding are linear layers (sometimes)
                no_decay.add(fpn)
            elif pn.endswith("weight") and isinstance(
                m, whitelist_weight_modules
            ):
                if (
                    "embedding" not in mn
                ):  # module named state_embedding is linear excluded.
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
            elif "W_" in pn:
                # weights of all tlens weight modules will be decayed
                decay.add(fpn)
            elif pn.endswith("weight") and isinstance(
                m, blacklist_weight_modules
            ):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)
            elif "ln1" in mn or "ln2" in mn or "ln_final" in mn:
                no_decay.add(fpn)  # layer norm weights will not be decayed

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert (
        len(inter_params) == 0
    ), "parameters %s made it into both decay/no_decay sets!" % (
        str(inter_params),
    )
    assert (
        len(param_dict.keys() - union_params) == 0
    ), "parameters %s were not separated into either decay/no_decay set!" % (
        str(param_dict.keys() - union_params),
    )

    # create the pytorch optimizer object
    optim_groups = [
        {
            "params": [param_dict[pn] for pn in sorted(list(decay))],
            "weight_decay": offline_config.weight_decay,
        },
        {
            "params": [param_dict[pn] for pn in sorted(list(no_decay))],
            "weight_decay": 0.0,
        },
    ]

    return optim_groups


def get_optimizer(
    optimizer_name: str,
    optim_groups: list[dict[str, Any]],
    lr: float,
    **kwargs,
):
    if optimizer_name.lower() == "sgd":
        return optim.SGD(optim_groups, lr=lr, **kwargs)
    elif optimizer_name.lower() == "adam":
        return optim.Adam(optim_groups, lr=lr, **kwargs)
    elif optimizer_name.lower() == "adamw":
        return optim.AdamW(optim_groups, lr=lr, **kwargs)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")


#  None
#  Linear Warmup and decay
#  Cosine Annealing with Warmup
#  Cosine Annealing with Warmup / Restarts
def get_scheduler(
    scheduler_name: Optional[str], optimizer: optim.Optimizer, **kwargs
):
    """
    Loosely based on this, seemed simpler write this than import
    transformers: https://huggingface.co/docs/transformers/main_classes/optimizer_schedules

    Args:
        scheduler_name (Optional[str]): Name of the scheduler to use. If None, returns a constant scheduler
        optimizer (optim.Optimizer): Optimizer to use
        **kwargs: Additional arguments to pass to the scheduler including warm_up_steps,
            training_steps, num_cycles, lr_end.
    """

    def get_warmup_lambda(warm_up_steps, training_steps):
        def lr_lambda(steps):
            if steps < warm_up_steps:
                return (steps + 1) / warm_up_steps
            else:
                return (training_steps - steps) / (
                    training_steps - warm_up_steps
                )

        return lr_lambda

    # heavily derived from hugging face although copilot helped.
    def get_warmup_cosine_lambda(warm_up_steps, training_steps, lr_end):
        def lr_lambda(steps):
            if steps < warm_up_steps:
                return (steps + 1) / warm_up_steps
            else:
                progress = (steps - warm_up_steps) / (
                    training_steps - warm_up_steps
                )
                return lr_end + 0.5 * (1 - lr_end) * (
                    1 + math.cos(math.pi * progress)
                )

        return lr_lambda

    if scheduler_name is None or scheduler_name.lower() == "constant":
        return lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda steps: 1.0)
    elif scheduler_name.lower() == "constantwithwarmup":
        warm_up_steps = kwargs.get("warm_up_steps", 0)
        return lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda steps: min(1.0, (steps + 1) / warm_up_steps),
        )
    elif scheduler_name.lower() == "linearwarmupdecay":
        warm_up_steps = kwargs.get("warm_up_steps", 0)
        training_steps = kwargs.get("training_steps")
        lr_lambda = get_warmup_lambda(warm_up_steps, training_steps)
        return lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif scheduler_name.lower() == "cosineannealing":
        training_steps = kwargs.get("training_steps")
        eta_min = kwargs.get("lr_end", 0)
        return lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=training_steps, eta_min=eta_min
        )
    elif scheduler_name.lower() == "cosineannealingwarmup":
        warm_up_steps = kwargs.get("warm_up_steps", 0)
        training_steps = kwargs.get("training_steps")
        eta_min = kwargs.get("lr_end", 0)
        lr_lambda = get_warmup_cosine_lambda(
            warm_up_steps, training_steps, eta_min
        )
        return lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif scheduler_name.lower() == "cosineannealingwarmrestarts":
        training_steps = kwargs.get("training_steps")
        eta_min = kwargs.get("lr_end", 0)
        num_cycles = kwargs.get("num_cycles", 1)
        T_0 = training_steps // num_cycles
        return lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=T_0, eta_min=eta_min
        )
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")
