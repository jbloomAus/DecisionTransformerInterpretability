import numpy as np
import torch
import torch.nn as nn
from positional_encodings.torch_encodings import PositionalEncoding2D, Summer
from torchtyping import TensorType as TT


class MiniGridBOWEmbedding(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        max_values=[11, 6, 3],
        channel_names=["object", "color", "state"],
        view_size=7,
        add_positional_enc=False,
    ):
        super().__init__()
        self.max_values = max_values
        self.max_value = max(max_values)
        self.channel_names = channel_names
        self.embedding_dim = embedding_dim
        self.view_size = view_size
        self.embedding = nn.Embedding(
            len(channel_names) * self.max_value, embedding_dim
        )
        self.position_encoding = (
            Summer(PositionalEncoding2D(embedding_dim))
            if add_positional_enc
            else nn.Identity()
        )

        # initialize embeddings
        initializer_range = 1.0 / np.sqrt(self.embedding_dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=initializer_range)

    def forward(self, inputs: TT["batch", "x", "y", "channel"]):  # noqa: F821
        offsets = torch.Tensor([0, self.max_value, 2 * self.max_value]).to(
            inputs.device
        )
        inputs = (inputs + offsets[None, None, None, :]).long()
        x = self.embedding(inputs).sum(3)
        x = self.position_encoding(x)
        return x

    def get_channel_embedding(self, channel_name):
        index = self.channel_names.index(channel_name)
        start_index = self.max_value * index
        end_index = self.max_value * (index) + self.max_values[index]
        full_embedding = self.embedding.weight[start_index:end_index]
        return full_embedding

    def get_all_channel_embeddings(self):
        """concat all channel embeddings"""
        return torch.cat(
            [
                self.get_channel_embedding(channel_name)
                for channel_name in self.channel_names
            ],
            dim=0,
        )

    def get_positional_encoding(self, inputs=None):
        if inputs is None:
            inputs = torch.zeros(
                (1, self.view_size, self.view_size, self.embedding_dim)
            )
        return self.position_encoding(inputs)
