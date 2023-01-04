from typing import Dict, Union

import einops
import numpy as np
import torch
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torchtyping import TensorType as TT
from transformer_lens import EasyTransformer, EasyTransformerConfig
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
seed_everything(42)

class StateEncoder(nn.Module):
    def __init__(self, n_embed):
        super(StateEncoder, self).__init__()
        self.n_embed = n_embed
        # input has shape 56 x 56 x 3
        # output has shape 1 x 1 x 512
        self.conv1 = nn.Conv2d(3, 32, 8, stride=4, padding=0) # 56 -> 13
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2, padding=0) # 13 -> 5
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1, padding=0) # 5 -> 3
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(576, n_embed)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        x = self.fc(x)
        x = F.relu(x)
        return x

class PosEmbedTokens(nn.Module):
    def __init__(self, cfg: Union[Dict, HookedTransformerConfig]):
        super().__init__()
        if isinstance(cfg, Dict):
            cfg = HookedTransformerConfig.from_dict(cfg)
        self.cfg = cfg
        self.W_pos = nn.Parameter(torch.empty(self.cfg.n_ctx, self.cfg.d_model))

    def forward(
        self, tokens: TT["batch", "position"], past_kv_pos_offset: int = 0
    ) -> TT["batch", "position", "d_model"]:
        """Tokens have shape [batch, pos]
        Output shape [pos, d_model] - will be broadcast along batch dim"""

        tokens_length = tokens.size(-2)
        pos_embed = self.W_pos[:tokens_length, :]  # [pos, d_model]
        broadcast_pos_embed = einops.repeat(
            pos_embed, "pos d_model -> batch pos d_model", batch=tokens.size(0)
        )  # [batch, pos, d_model]
        return broadcast_pos_embed

class DecisionTransformer(torch.nn.Module):
    def __init__(self, env, state_embedding_type: str = 'CNN', max_game_length: int = 1000):
        '''
        model = Classifier(cfg)
        '''
        super().__init__()

        self.env = env
        self.d_model = 64
        self.block_size = 10
        self.max_timestep = max_game_length
        self.state_embedding_type = state_embedding_type

        self.ctx_size = 10*self.block_size # we can handle 10 full timesteps

        # Embedding layers
        self.pos_emb = nn.Parameter(torch.zeros(1, self.ctx_size, self.d_model))
        self.time_embedding = nn.Embedding(self.max_timestep+1, self.d_model)

        if state_embedding_type == 'CNN':
            self.state_encoder = StateEncoder(self.d_model)
        else: 
            n_obs = np.prod(env.observation_space['image'].shape)
            self.state_encoder = nn.Linear(n_obs, self.d_model)
            nn.init.normal_(self.state_encoder.weight, mean=0.0, std=0.02)

        self.action_embedding = nn.Sequential(nn.Embedding(env.action_space.n + 1, self.d_model), nn.Tanh())
        self.reward_embedding = nn.Sequential(nn.Linear(1, self.d_model), nn.Tanh())

        # Initialize weights
        nn.init.normal_(self.action_embedding[0].weight, mean=0.0, std=0.02)
        nn.init.normal_(self.reward_embedding[0].weight, mean=0.0, std=0.02)

        # Transformer
        cfg = EasyTransformerConfig(
            n_layers=2,
            d_model=self.d_model,
            d_head=32,
            n_heads=2,
            d_mlp=128,
            d_vocab= 64,
            n_ctx= self.ctx_size,
            act_fn="relu",
            # normalization_type=None,
            attention_dir="causal",
            d_vocab_out=env.action_space.n, #
            seed = 0
        )

        assert cfg.attention_dir == "causal", "Attention direction must be causal"
        # assert cfg.normalization_type is None, "Normalization type must be None"

        self.transformer = EasyTransformer(cfg)

        # Because we passing in tokens, turn off embedding and update the position embedding
        self.transformer.embed = nn.Identity()
        self.transformer.pos_embed = PosEmbedTokens(cfg)

    # state, action, and return
    def forward(self, states, actions, targets=None, rtgs=None, timesteps=None):
        # states: (batch, block_size, 56, 56, 3)
        # actions: (batch, block_size, 1)
        # targets: (batch, block_size, 1)
        # rtgs: (batch, block_size, 1)
        # timesteps: (batch, 1, 1) # this seems wrong because the time should be different for each element in the each block (incrememnting by 1)

        # embed states and recast back to (batch, block_size, n_embd)
        state_embeddings = self.get_state_embeddings(states) # batch_size, block_size, n_embd
        action_embeddings = self.get_action_embeddings(actions) if actions is not None else None # batch_size, block_size, n_embd or None
        reward_embeddings = self.get_reward_embeddings(rtgs) # batch_size, block_size, n_embd
        time_embeddings = self.get_time_embeddings(timesteps) # batch_size, block_size, n_embd

        # use state_embeddings, actions, rewards to go and 
        token_embeddings = self.get_token_embeddings(
            state_embeddings = state_embeddings, 
            action_embeddings = action_embeddings,
            reward_embeddings = reward_embeddings,
            time_embeddings = time_embeddings
        )

        # use transformer to model sequence and return embeddings for states, actions, and rewards

        # since we are passing the transformer an embedding, we need to
        logits = self.transformer(token_embeddings)

        # print("logits: ", logits[0, 0, :10])

        # select hidden states for actions
        logits = self.get_logits(logits, actions)
        # if we are given some desired targets also calculate the loss
        loss = self.get_loss(logits, targets)

        return logits, loss

    def get_logits(self, logits, actions):
        if actions is not None:
            logits = logits[:, 1::3, :] # only keep predictions from state_embeddings (predictions over actions)
        elif actions is None:
            logits = logits[:, 1:, :]
        else:
            raise NotImplementedError()
        
        return logits

    def get_loss(self, logits, targets):
        if targets is not None:
            return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        else: 
            return None

    def check_input_sizes(self, states, actions, targets):
        assert states.shape[0] == actions.shape[0] == targets.shape[0], "batch sizes must be the same"
        assert states.shape[1] == actions.shape[1] == targets.shape[1], "block sizes must be the same"

    def get_time_embeddings(self, timesteps):

        assert timesteps.max() <= self.max_timestep, "timesteps must be less than max_timesteps"

        block_size = timesteps.shape[1]
        timesteps = rearrange(timesteps, 'batch block time-> (batch block) time')
        time_embeddings = self.time_embedding(timesteps).squeeze(-2)
        time_embeddings = rearrange(time_embeddings, '(batch block) n_embd -> batch block n_embd', block=block_size)
        return time_embeddings

    def get_state_embeddings(self, states):
        # embed states and recast back to (batch, block_size, n_embd)
        block_size = states.shape[1]
        if self.state_embedding_type == "CNN":
            states = rearrange(states, 'batch block height width channel -> (batch block) channel height width')
            state_embeddings = self.state_encoder(states.type(torch.float32).contiguous()) # (batch * block_size, n_embd)
        else: 
            states = rearrange(states, 'batch block height width channel -> (batch block) (channel height width)')
            state_embeddings = self.state_encoder(states.type(torch.float32).contiguous()) # (batch * block_size, n_embd)
        state_embeddings = rearrange(state_embeddings, '(batch block) n_embd -> batch block n_embd', block=block_size)
        return state_embeddings

    def get_action_embeddings(self, actions):
        # embed actions
        block_size = actions.shape[1]
        actions = rearrange(actions, 'batch block action -> (batch block) action')
        action_embeddings = self.action_embedding(actions).flatten(1) # I don't see why we need this but we do? Maybe because of the sequential?
        action_embeddings = rearrange(action_embeddings, '(batch block) n_embd -> batch block n_embd', block=block_size)
        return action_embeddings

    def get_reward_embeddings(self, rtgs):
        block_size = rtgs.shape[1]
        rtgs = rearrange(rtgs, 'batch block rtg -> (batch block) rtg')
        rtg_embeddings = self.reward_embedding(rtgs.type(torch.float32))
        rtg_embeddings = rearrange(rtg_embeddings, '(batch block) n_embd -> batch block n_embd', block=block_size)
        return rtg_embeddings

    def get_token_embeddings(self, 
        state_embeddings, 
        time_embeddings, 
        action_embeddings, 
        reward_embeddings, 
        targets = None):
        '''
        We need to compose the embeddings for:
            - states
            - actions
            - rewards
            - time

        Handling the cases where:
        1. we are training:
            1. we may not have action yet (reward, state)
            2. we have (action, state, reward)...
        2. we are evaluating:
            1. we have a target "a reward" followed by state
        
        1.1 and 2.1 are the same, but we need to handle the target as the initial reward.
        
        '''
        batches = state_embeddings.shape[0]
        block_size = state_embeddings.shape[1]

        reward_embeddings = reward_embeddings + time_embeddings
        state_embeddings = state_embeddings   + time_embeddings
        
        if action_embeddings is not None:
            action_embeddings = action_embeddings + time_embeddings
        if targets:
            targets = targets + time_embeddings

        # estimate the trajectory length:
        # 1. if we have actions, we have 3 tokens per timestep
        # 2. if we don't have actions, we have 2 tokens per timestep (and one time step)
        timesteps = time_embeddings.shape[1] # number of timesteps
        if action_embeddings is not None:
            trajectory_length = timesteps*3 
        else:
            trajectory_length = 2 # one timestep, no action yet

        # create the token embeddings
        token_embeddings = torch.zeros(
            (batches, trajectory_length, self.d_model),  
            dtype=torch.float32, device=state_embeddings.device) # batches, blocksize, n_embd
 
        if action_embeddings is not None:
            token_embeddings[:,::3,:] = reward_embeddings
            token_embeddings[:,1::3,:] = state_embeddings 
            token_embeddings[:,2::3,:] = action_embeddings
        else:
            token_embeddings[:,0,:] = reward_embeddings[:,0,:]
            token_embeddings[:,1,:] = state_embeddings[:,0,:]

        if targets is not None:
            token_embeddings[:,0,:] = targets[:,0,:]

        if trajectory_length > self.transformer.cfg.n_ctx:
            raise ValueError("Trajectory length is greater than the maximum sequence length for this model")
            # or we could truncate, deal with this later...

        return token_embeddings