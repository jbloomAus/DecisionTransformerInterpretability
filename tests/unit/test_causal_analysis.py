import pytest
import torch
from copy import deepcopy
from transformer_lens.hook_points import HookPoint
from src.streamlit_app.causal_analysis_components import get_ablation_function


@pytest.fixture
def hook():
    return HookPoint()

def test_get_ablation_function_head(hook):
    hook.remove_hooks("both")
    ablation_func = get_ablation_function(
        ablate_to_mean=False, head_to_ablate=0, component="HEAD"
    )
    x = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)[
        None, None, None, :
    ]  # shape (1, 1, 1, 5)
    non_ablated_value = hook(x)
    hook.add_hook(ablation_func)
    ablated_value = hook(deepcopy(x))
    assert non_ablated_value.mean() == 3.0
    assert ablated_value.mean() == 0.0

    hook.remove_hooks("both")
    ablation_func = get_ablation_function(
        ablate_to_mean=True, head_to_ablate=0, component="HEAD"
    )
    x = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)[
        None, None, None, :
    ]  # shape (1, 1, 1, 5)
    non_ablated_value = hook(x)
    hook.add_hook(ablation_func)
    ablated_value = hook(deepcopy(x))
    assert non_ablated_value.mean() == 3.0
    assert ablated_value[0, 0, 0, 0] == 3.0


def test_get_ablation_function_mlp(hook):
    hook.remove_hooks("both")
    ablation_func = get_ablation_function(
        ablate_to_mean=False, head_to_ablate=0, component="MLP"
    )
    x = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)[
        None, None, :
    ]  # shape (1, 1, 5)
    non_ablated_value = hook(x)
    hook.add_hook(ablation_func)
    ablated_value = hook(deepcopy(x))
    assert non_ablated_value.mean() == 3.0
    assert ablated_value.mean() == 0.0

    hook.remove_hooks("both")
    ablation_func = get_ablation_function(
        ablate_to_mean=True, head_to_ablate=0, component="MLP"
    )
    x = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)[
        None, None, :
    ]  # shape (1, 1, 5)
    non_ablated_value = hook(x)
    hook.add_hook(ablation_func)
    ablated_value = hook(deepcopy(x))
    assert non_ablated_value.mean() == 3.0
    assert ablated_value[0, 0, 0] == 3.0
