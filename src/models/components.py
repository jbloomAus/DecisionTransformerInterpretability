from typing import Dict, Union

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from positional_encodings.torch_encodings import PositionalEncoding2D, Summer
from torchtyping import TensorType
from torchtyping import TensorType as TT
from transformer_lens import HookedTransformer, HookedTransformerConfig


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
        x = x.permute(0, 3, 1, 2)
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


# Taken from BabyAI: Duplicate in this repo as I don't want to touch the trajectory LSTM one.
# From https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def initialize_parameters(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(
            m.weight.data.pow(2).sum(1, keepdim=True)
        )
        if m.bias is not None:
            m.bias.data.fill_(0)


class MiniGridConvEmbedder(nn.Module):
    """
    Equivalent to BabyAI state embedding with use_instruction = False.

    Set endpool=True:
        If you want to preserve the spatial resolution of the feature maps after processing
        through the convolutional layers. This may be helpful if the input images have a smaller
        size or if you need to retain more spatial information for downstream tasks.

    Set endpool=False:
        If you want to reduce the spatial dimensions of the output feature maps by applying max-pooling.
        This can help in aggregating local information and reducing the number of parameters in the network,
        which may lead to faster training and potentially better generalization.
    """

    def __init__(self, embedding_dim: int = 128, endpool: bool = False):
        super(MiniGridConvEmbedder, self).__init__()

        # TODO: pass on args to enable variable inputs (currently by default)
        # it will support 7*7*3 inputs
        self.image_bow = MiniGridBOWEmbedding(embedding_dim)
        self.film_pool = nn.MaxPool2d(
            kernel_size=(7, 7) if endpool else (2, 2), stride=2
        )
        self.conv1 = nn.Conv2d(
            in_channels=embedding_dim,
            out_channels=embedding_dim,
            kernel_size=(3, 3) if endpool else (2, 2),
            stride=1,
            padding=1,
        )
        self.bn1 = nn.BatchNorm2d(embedding_dim)
        self.relu1 = nn.ReLU()
        self.pool1 = (
            nn.Identity()
            if endpool
            else nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        )
        self.conv2 = nn.Conv2d(
            in_channels=embedding_dim,
            out_channels=embedding_dim,
            kernel_size=(3, 3),
            padding=1,
        )
        self.bn2 = nn.BatchNorm2d(embedding_dim)
        self.relu2 = nn.ReLU()
        self.pool2 = (
            nn.Identity()
            if endpool
            else nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        )
        self.apply(
            initialize_parameters
        )  # use default initialization of BabyAI

    def forward(
        self,
        x: TensorType["batch", "height", "width", "channel"],  # noqa: F821
    ) -> TensorType["batch", "embedding_dim", "height", "width"]:  # noqa: F821
        x = self.image_bow(x)  # Shape: [batch, embedding_dim, height, width]
        x = self.conv1(
            x
        )  # Shape: [batch, embedding_dim, height, width] (if endpool=True), [batch, embedding_dim, (height-1)/2+1, (width-1)/2+1] (if endpool=False)
        x = self.bn1(x)  # Shape: same as above
        x = self.relu1(x)  # Shape: same as above
        x = self.pool1(
            x
        )  # Shape: same as above (if endpool=True), [batch, embedding_dim, height//2, width//2] (if endpool=False)
        x = self.conv2(
            x
        )  # Shape: [batch, embedding_dim, height, width] (if endpool=True), [batch, embedding_dim, height//2, width//2] (if endpool=False)
        x = self.bn2(x)  # Shape: same as above
        x = self.relu2(x)  # Shape: same as above
        x = self.pool2(
            x
        )  # Shape: same as above (if endpool=True), [batch, embedding_dim, (height//2)//2, (width//2)//2] (if endpool=False)
        x = F.relu(self.film_pool(x))
        x = x.reshape(x.shape[0], -1)
        return x


class MiniGridViTEmbedder(nn.Module):
    """
    Experimental use of decoder only transformer as a vision model.

    Differs from tradition ViT in that:
    - It's not an encoder.
    - We'll use the last token not the first.
    - Will use positional encoding
    """

    def __init__(self, embedding_dim: int = 128):
        super(MiniGridViTEmbedder, self).__init__()

        self.image_bow = MiniGridBOWEmbedding(
            embedding_dim=embedding_dim,
            max_values=[11, 6, 3],
            channel_names=["object", "color", "state"],
            view_size=7,
            add_positional_enc=True,
        )

        n_heads = 4
        d_head = embedding_dim // n_heads
        vit_config = HookedTransformerConfig(
            n_layers=1,
            d_model=embedding_dim,
            d_head=d_head,
            n_heads=n_heads,  #
            d_mlp=256,  # doesn't matter, turning off
            d_vocab=128,  # needs to match the model.
            n_ctx=7 * 7,  # 7x7 grid
            normalization_type=None,
            attention_dir="causal",  # this is usually bidirectional
            attn_only=True,
            d_vocab_out=embedding_dim,  # adds a linear layer to the output
        )

        transformer = HookedTransformer(vit_config)
        # we're going to pass it tokens via BOW.
        transformer.embed = nn.Identity()
        transformer.pos_embed = PosEmbedTokens(vit_config)
        nn.init.normal_(
            transformer.pos_embed.W_pos, vit_config.initializer_range
        )
        self.transformer = transformer

    def forward(
        self,
        x: TensorType["batch", "height", "width", "channel"],  # noqa: F821
    ) -> TensorType["batch", "embedding_dim", "height", "width"]:  # noqa: F821
        x = self.image_bow(x)  # Shape: [batch, embedding_dim, height, width]
        x = x.flatten(-2, -1).permute(
            0, 2, 1
        )  # Shape: [batch, embedding_dim, height*width]
        x = self.transformer(x)  # Shape: [batch, height*width, embedding_dim]
        return x[:, -1, :]  # Shape: [batch, embedding_dim] (last token)


class PosEmbedTokens(nn.Module):
    def __init__(self, cfg: Union[Dict, HookedTransformerConfig]):
        super().__init__()
        if isinstance(cfg, Dict):
            cfg = HookedTransformerConfig.from_dict(cfg)
        self.cfg = cfg
        self.W_pos = nn.Parameter(
            torch.empty(self.cfg.n_ctx, self.cfg.d_model)
        )

    def forward(
        self,
        tokens: TT["batch", "position"],  # noqa: F821
        past_kv_pos_offset: int = 0,
    ) -> TT["batch", "position", "d_model"]:  # noqa: F821
        """Tokens have shape [batch, pos]
        Output shape [pos, d_model] - will be broadcast along batch dim"""

        tokens_length = tokens.size(-2)
        pos_embed = self.W_pos[:tokens_length, :]  # [pos, d_model]
        broadcast_pos_embed = einops.repeat(
            pos_embed, "pos d_model -> batch pos d_model", batch=tokens.size(0)
        )  # [batch, pos, d_model]

        return broadcast_pos_embed
