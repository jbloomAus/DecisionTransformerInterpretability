"""
This model is a modified version of the original model from the BabyAI repository.
https://github.com/kimiyoung/transformer-xl/blob/44781ed21dbaec88b280f74d9ae2877f52b492a5/pytorch/mem_transformer.py

I'm not going to keep track of the changes since the original code is not easily runnable.
Hopefully we can get the code working and work off that as a benchmark.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.config import EnvironmentConfig, LSTMModelConfig

# This is from BabyAI, not sure why we need it yet.
# dictionary that defines what head is required for each extra info used for auxiliary supervision
required_heads = {
    "seen_state": "binary",
    "see_door": "binary",
    "see_obj": "binary",
    "obj_in_instr": "binary",
    # multi class classifier with 9 possible classes
    "in_front_of_what": "multiclass9",
    # continous regressor with outputs in [0, 1]
    "visit_proportion": "continuous01",
    "bot_action": "binary",
}

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


# Inspired by FiLMedBlock from https://arxiv.org/abs/1709.07871
class FiLM(nn.Module):
    def __init__(self, in_features, out_features, in_channels, imm_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=imm_channels,
            kernel_size=(3, 3),
            padding=1,
        )
        self.bn1 = nn.BatchNorm2d(imm_channels)
        self.conv2 = nn.Conv2d(
            in_channels=imm_channels,
            out_channels=out_features,
            kernel_size=(3, 3),
            padding=1,
        )
        self.bn2 = nn.BatchNorm2d(out_features)

        self.weight = nn.Linear(in_features, out_features)
        self.bias = nn.Linear(in_features, out_features)

        self.apply(initialize_parameters)

    def forward(self, x, y):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        weight = self.weight(y).unsqueeze(2).unsqueeze(3)
        bias = self.bias(y).unsqueeze(2).unsqueeze(3)
        out = x * weight + bias
        return F.relu(self.bn2(out))


class ImageBOWEmbedding(nn.Module):
    def __init__(self, observation_space, embedding_dim):
        super().__init__()
        self.max_value = self.get_max_value(observation_space)
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(3 * self.max_value, embedding_dim)
        self.apply(initialize_parameters)

    def forward(self, inputs):
        offsets = torch.Tensor([0, self.max_value, 2 * self.max_value]).to(
            inputs.device
        )
        inputs = (inputs + offsets[None, :, None, None]).long()
        return self.embedding(inputs).sum(1).permute(0, 3, 1, 2)

    def get_max_value(self, observation_space):
        if hasattr(observation_space, "max_value"):
            return observation_space.max_value
        else:
            return 9  # default value for BabyAI (compact encoding default)


class TrajectoryLSTM(nn.Module):
    def __init__(
        self,
        model_config: LSTMModelConfig
        #  observation_space,
        #  action_space,
        #  image_dim=128,
        #  memory_dim=128,
        #  instr_dim=128,
        #  use_instr=False,
        #  lang_model="gru",
        #  use_memory=False,
        #  arch="bow_endpool_res",
        #  aux_info=None
    ):
        """
        A model for actor-critic with options for different architectures and auxiliary tasks.

        Inherits from:
            nn.Module: Base class for all neural network modules in PyTorch.
            RecurrentACModel: Custom class for recurrent actor-critic models.

        Args:
            observation_space: A dictionary containing the observation space of the environment.
            action_space: The action space of the environment.
            image_dim: The dimension of the image embedding. Default is 128.
            memory_dim: The dimension of the memory embedding. Default is 128.
            instr_dim: The dimension of the instruction embedding. Default is 128.
            use_instr: A boolean flag to indicate whether to use instructions or not. Default is False.
            lang_model: A string specifying the language model to be used for instructions. Default is "gru".
            use_memory: A boolean flag to indicate whether to use memory or not. Default is False.
            arch: A string specifying the architecture to be used. Default is "bow_endpool_res".
            aux_info: Additional information to be used for auxiliary tasks. Default is None.
        """
        super().__init__()

        self.endpool = model_config.endpool
        self.bow = model_config.bow
        self.pixel = model_config.pixel
        self.res = model_config.res

        # Decide which components are enabled
        self.use_instr = model_config.use_instr
        self.use_memory = model_config.use_memory
        self.arch = model_config.arch
        self.lang_model = model_config.lang_model
        self.aux_info = model_config.aux_info
        self.image_dim = model_config.image_dim
        self.memory_dim = model_config.memory_dim
        self.instr_dim = model_config.instr_dim
        self.observation_space = (
            model_config.environment_config.observation_space
        )
        self.action_space = model_config.environment_config.action_space

        for part in self.arch.split("_"):
            if part not in [
                "original",
                "bow",
                "pixels",
                "endpool",
                "res",
                "simple",
            ]:
                raise ValueError(
                    "Incorrect architecture name: {}".format(self.arch)
                )

        # if not self.use_instr:
        #     raise ValueError("FiLM architecture can be used when instructions are enabled")
        self.image_conv = nn.Sequential(
            *[
                *(
                    [ImageBOWEmbedding(self.observation_space["image"], 128)]
                    if self.bow
                    else []
                ),
                *(
                    [
                        nn.Conv2d(
                            in_channels=3,
                            out_channels=128,
                            kernel_size=(8, 8),
                            stride=8,
                            padding=0,
                        )
                    ]
                    if self.pixel
                    else []
                ),
                nn.Conv2d(
                    in_channels=128 if self.bow or self.pixel else 3,
                    out_channels=128,
                    kernel_size=(3, 3) if self.endpool else (2, 2),
                    stride=1,
                    padding=1,
                ),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                *(
                    []
                    if self.endpool
                    else [nn.MaxPool2d(kernel_size=(2, 2), stride=2)]
                ),
                nn.Conv2d(
                    in_channels=128,
                    out_channels=128,
                    kernel_size=(3, 3),
                    padding=1,
                ),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                *(
                    []
                    if self.endpool
                    else [nn.MaxPool2d(kernel_size=(2, 2), stride=2)]
                ),
            ]
        )
        self.film_pool = nn.MaxPool2d(
            kernel_size=(7, 7) if self.endpool else (2, 2), stride=2
        )

        # Define instruction embedding
        if self.use_instr:
            if self.lang_model in ["gru", "bigru", "attgru"]:
                self.word_embedding = nn.Embedding(
                    self.observation_space["mission"][0].n + 1, self.instr_dim
                )
                if self.lang_model in ["gru", "bigru", "attgru"]:
                    gru_dim = self.instr_dim
                    if self.lang_model in ["bigru", "attgru"]:
                        gru_dim //= 2
                    self.instr_rnn = nn.GRU(
                        self.instr_dim,
                        gru_dim,
                        batch_first=True,
                        bidirectional=(self.lang_model in ["bigru", "attgru"]),
                    )
                    self.final_instr_dim = self.instr_dim
                else:
                    kernel_dim = 64
                    kernel_sizes = [3, 4]
                    self.instr_convs = nn.ModuleList(
                        [
                            nn.Conv2d(1, kernel_dim, (K, self.instr_dim))
                            for K in kernel_sizes
                        ]
                    )
                    self.final_instr_dim = kernel_dim * len(kernel_sizes)

            if self.lang_model == "attgru":
                self.memory2key = nn.Linear(
                    self.memory_size, self.final_instr_dim
                )

            num_module = 2
            self.controllers = []
            for ni in range(num_module):
                mod = FiLM(
                    in_features=self.final_instr_dim,
                    out_features=128
                    if ni < num_module - 1
                    else self.image_dim,
                    in_channels=128,
                    imm_channels=128,
                )
                self.controllers.append(mod)
                self.add_module("FiLM_" + str(ni), mod)

        # Define memory and resize image embedding
        self.embedding_size = self.image_dim
        if self.use_memory:
            self.memory_rnn = nn.LSTMCell(self.image_dim, self.memory_dim)
            self.embedding_size = self.semi_memory_size

        # Define actor's model
        self.actor = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, self.action_space.n),
        )

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size, 64), nn.Tanh(), nn.Linear(64, 1)
        )

        # Initialize parameters correctly
        self.apply(initialize_parameters)

        # Define head for extra info
        if self.aux_info:
            self.extra_heads = None
            self.add_heads()

        # if simple in arch, we just want to embedding using a simple embedding (no CNN)
        if "simple" in self.arch:
            # if it's a continuous embeding just flatten and project.
            self.simple_embedding = nn.Linear(
                np.prod(self.observation_space.shape).astype(int),
                self.image_dim,
            )
        self = self.to(model_config.device)

    def add_heads(self):
        """
        When using auxiliary tasks, the environment yields at each step some binary, continous, or multiclass
        information. The agent needs to predict those information. This function add extra heads to the model
        that output the predictions. There is a head per extra information (the head type depends on the extra
        information type).
        """
        self.extra_heads = nn.ModuleDict()
        for info in self.aux_info:
            if required_heads[info] == "binary":
                self.extra_heads[info] = nn.Linear(self.embedding_size, 1)
            elif required_heads[info].startswith("multiclass"):
                n_classes = int(required_heads[info].split("multiclass")[-1])
                self.extra_heads[info] = nn.Linear(
                    self.embedding_size, n_classes
                )
            elif required_heads[info].startswith("continuous"):
                if required_heads[info].endswith("01"):
                    self.extra_heads[info] = nn.Sequential(
                        nn.Linear(self.embedding_size, 1), nn.Sigmoid()
                    )
                else:
                    raise ValueError("Only continous01 is implemented")
            else:
                raise ValueError("Type not supported")
            # initializing these parameters independently is done in order to have consistency of results when using
            # supervised-loss-coef = 0 and when not using any extra binary information
            self.extra_heads[info].apply(initialize_parameters)

    def add_extra_heads_if_necessary(self, aux_info):
        """
        This function allows using a pre-trained model without aux_info and add aux_info to it and still make
        it possible to finetune.
        """
        try:
            if not hasattr(self, "aux_info") or not set(self.aux_info) == set(
                aux_info
            ):
                self.aux_info = aux_info
                self.add_heads()
        except Exception:
            raise ValueError("Could not add extra heads")

    @property
    def memory_size(self):
        return 2 * self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.memory_dim

    def forward(self, obs, memory, instr_embedding=None):
        if self.use_instr and instr_embedding is None:
            instr_embedding = self._get_instr_embedding(obs.mission)
        if self.use_instr and self.lang_model == "attgru":
            # outputs: B x L x D
            # memory: B x M
            mask = (obs.mission != 0).float()
            # The mask tensor has the same length as obs.instr, and
            # thus can be both shorter and longer than instr_embedding.
            # It can be longer if instr_embedding is computed
            # for a subbatch of obs.instr.
            # It can be shorter if obs.instr is a subbatch of
            # the batch that instr_embeddings was computed for.
            # Here, we make sure that mask and instr_embeddings
            # have equal length along dimension 1.
            mask = mask[:, : instr_embedding.shape[1]]
            instr_embedding = instr_embedding[:, : mask.shape[1]]

            keys = self.memory2key(memory)
            pre_softmax = (keys[:, None, :] * instr_embedding).sum(
                2
            ) + 1000 * mask
            attention = F.softmax(pre_softmax, dim=1)
            instr_embedding = (instr_embedding * attention[:, :, None]).sum(1)

        if "simple" not in self.arch:  # batch of 2D images
            x = torch.transpose(torch.transpose(obs.image, 1, 3), 2, 3)

            if "pixel" in self.arch:
                x /= 256.0
            x = self.image_conv(x)
            if self.use_instr:
                for controller in self.controllers:
                    out = controller(x, instr_embedding)
                    if self.res:
                        out += x
                    x = out

            x = F.relu(self.film_pool(x))
            x = x.reshape(x.shape[0], -1)
        else:
            x = self.simple_embedding(obs.image)

        if self.use_memory:
            hidden = (
                memory[:, : self.semi_memory_size],
                memory[:, self.semi_memory_size :],
            )
            hidden = self.memory_rnn(x, hidden)
            embedding = hidden[0]
            memory = torch.cat(hidden, dim=1)
        else:
            embedding = x

        if hasattr(self, "aux_info") and self.aux_info:
            extra_predictions = {
                info: self.extra_heads[info](embedding)
                for info in self.extra_heads
            }
        else:
            extra_predictions = dict()

        x = self.actor(embedding)
        # check if x is made of nans
        if torch.isnan(x).any():
            raise ValueError("x is made of nans")
        dist = Categorical(logits=F.log_softmax(x, dim=1))

        x = self.critic(embedding)
        value = x.squeeze(1)

        return {
            "dist": dist,
            "value": value,
            "memory": memory,
            "extra_predictions": extra_predictions,
        }

    def _get_instr_embedding(self, instr):
        lengths = (instr != 0).sum(1).long()
        if self.lang_model == "gru":
            out, _ = self.instr_rnn(self.word_embedding(instr))
            hidden = out[range(len(lengths)), lengths - 1, :]
            return hidden

        elif self.lang_model in ["bigru", "attgru"]:
            masks = (instr != 0).float()

            if lengths.shape[0] > 1:
                seq_lengths, perm_idx = lengths.sort(0, descending=True)
                iperm_idx = torch.LongTensor(perm_idx.shape).fill_(0)
                if instr.is_cuda:
                    iperm_idx = iperm_idx.cuda()
                for i, v in enumerate(perm_idx):
                    iperm_idx[v.data] = i

                inputs = self.word_embedding(instr)
                inputs = inputs[perm_idx]

                inputs = pack_padded_sequence(
                    inputs, seq_lengths.data.cpu().numpy(), batch_first=True
                )

                outputs, final_states = self.instr_rnn(inputs)
            else:
                instr = instr[:, 0 : lengths[0]]
                outputs, final_states = self.instr_rnn(
                    self.word_embedding(instr)
                )
                iperm_idx = None
            final_states = final_states.transpose(0, 1).contiguous()
            final_states = final_states.view(final_states.shape[0], -1)
            if iperm_idx is not None:
                outputs, _ = pad_packed_sequence(outputs, batch_first=True)
                outputs = outputs[iperm_idx]
                final_states = final_states[iperm_idx]

            return outputs if self.lang_model == "attgru" else final_states

        else:
            ValueError(
                "Undefined instruction architecture: {}".format(self.use_instr)
            )
