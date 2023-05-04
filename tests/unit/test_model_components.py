import pytest
import torch
import torch.nn as nn
from src.models.components import (
    MiniGridBOWEmbedding,
    MiniGridConvEmbedder,
    MiniGridViTEmbedder,
)


from src.environments.memory import MemoryEnv
from minigrid.wrappers import ViewSizeWrapper
from positional_encodings.torch_encodings import Summer


@pytest.fixture
def env():
    env = ViewSizeWrapper(
        MemoryEnv(
            size=7,
            random_length=False,
            random_start_pos=False,
            max_steps=200,
            render_mode="rgb_array",
        ),
        7,
    )
    return env


@pytest.fixture
def obs(env):
    obs, info = env.reset()
    obs = torch.from_numpy(obs["image"]).unsqueeze(0)
    return obs


# TODO: Test norms later.
# def validate_dist_of_layers(model, expected_std = 0.03):

#     # Iterate through the submodules (layers) and check the norm
#     for name, module in model.named_modules():
#         if isinstance(module, (nn.Linear, nn.Conv2d, nn.BatchNorm2d)):
#             weight_mean= torch.mean(module.weight.data)
#             print(f"Layer: {name}, Mean: {weight_mean}")
#             assert weight_mean == pytest.approx(0,0, abs=1e-3)

#             weight_std = torch.std(module.weight.data)
#             print(f"Layer: {name}, Std: {weight_std}")
#             assert weight_std == pytest.approx(expected_std, abs=1e-2)

#             if module.bias is not None:
#                 bias_mean = torch.mean(module.bias.data)
#                 print(f"Layer: {name}, Bias Mean: {bias_mean}")
#                 assert bias_mean == pytest.approx(0.0, abs=1e-3)

#                 bias_std = torch.std(module.bias.data)
#                 print(f"Layer: {name}, Bias Std: {bias_std}")
#                 assert bias_std == pytest.approx(expected_std, abs=1e-2)


def test_MiniGridBOWEmbedding_standard(obs):
    state_embedding = MiniGridBOWEmbedding(
        embedding_dim=32,
        max_values=[11, 6, 3],
        channel_names=["object", "color", "state"],
        view_size=7,
        add_positional_enc=True,
    )

    embed_2d = state_embedding(obs).detach()

    assert embed_2d.shape == (1, 32, 7, 7)

    # check each channel shape:
    assert state_embedding.get_channel_embedding("object").shape == (11, 32)
    assert state_embedding.get_channel_embedding("color").shape == (6, 32)
    assert state_embedding.get_channel_embedding("state").shape == (3, 32)
    assert state_embedding.get_all_channel_embeddings().shape == (20, 32)

    # assert norm
    all_emb = state_embedding.get_all_channel_embeddings().detach()
    assert torch.norm(all_emb, dim=-1).mean() == pytest.approx(1.0, 0.1)

    # check positional encoding shape:
    assert state_embedding.get_positional_encoding().shape == (1, 7, 7, 32)

    # check norm of positional encoding:
    assert (
        torch.norm(state_embedding.get_positional_encoding(), dim=3).mean()
        > 0.5
    )

    # assert the layer position encoding is not the identity
    assert isinstance(state_embedding.position_encoding, Summer)


def test_MiniGridBOWEmbedding_no_position(obs):
    state_embedding = MiniGridBOWEmbedding(
        embedding_dim=32,
        max_values=[11, 6, 3],
        channel_names=["object", "color", "state"],
        view_size=7,
        add_positional_enc=False,
    )

    embed_2d = state_embedding(obs).detach()

    assert embed_2d.shape == (1, 32, 7, 7)

    # assert norm
    all_emb = state_embedding.get_all_channel_embeddings().detach()
    assert torch.norm(all_emb, dim=-1).mean() == pytest.approx(1.0, 0.1)

    # check each channel shape:
    assert state_embedding.get_channel_embedding("object").shape == (11, 32)
    assert state_embedding.get_channel_embedding("color").shape == (6, 32)
    assert state_embedding.get_channel_embedding("state").shape == (3, 32)
    assert state_embedding.get_all_channel_embeddings().shape == (20, 32)

    # check positional encoding shape:
    assert state_embedding.get_positional_encoding().shape == (1, 7, 7, 32)

    # assert the layer position encoding is the identity
    assert isinstance(state_embedding.position_encoding, torch.nn.Identity)


def test_MiniGridConvEmbedder(obs):
    image_conv = MiniGridConvEmbedder(
        embedding_dim=32,
        endpool=True,
    )

    embed_2d = image_conv(obs).detach()
    assert embed_2d.shape == (1, 32)
    # validate_dist_of_layers(image_conv)


def test_MiniGridConvEmbedder_no_endpool(obs):
    image_conv = MiniGridConvEmbedder(
        embedding_dim=32,
        endpool=False,
    )

    embed_2d = image_conv(obs).detach()
    assert embed_2d.shape == (1, 32)

    num_params = sum(p.numel() for p in image_conv.parameters())
    assert num_params == 14560  # that's insane!


def test_MiniGridViTEmbedder(obs):
    image_conv = MiniGridViTEmbedder(embedding_dim=32)

    embed_2d = image_conv(obs).detach()
    assert embed_2d.shape == (1, 32)

    num_params = sum(p.numel() for p in image_conv.parameters())
    assert num_params == 7904  # that's closer to being reasonable.
