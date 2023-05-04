import pytest
import torch
from src.models.components import MiniGridBOWEmbedding


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


def test_MiniGridBOWEmbedding_standard(env):
    state_embedding = MiniGridBOWEmbedding(
        embedding_dim=32,
        max_values=[11, 6, 3],
        channel_names=["object", "color", "state"],
        view_size=7,
        add_positional_enc=True,
    )

    obs, info = env.reset()
    obs = torch.from_numpy(obs["image"]).unsqueeze(0)
    embed_2d = state_embedding(obs).detach()

    assert embed_2d.shape == (1, 7, 7, 32)

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


def test_MiniGridBOWEmbedding_no_position(env):
    state_embedding = MiniGridBOWEmbedding(
        embedding_dim=32,
        max_values=[11, 6, 3],
        channel_names=["object", "color", "state"],
        view_size=7,
        add_positional_enc=False,
    )

    obs, info = env.reset()
    obs = torch.from_numpy(obs["image"]).unsqueeze(0)
    embed_2d = state_embedding(obs).detach()

    assert embed_2d.shape == (1, 7, 7, 32)

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
