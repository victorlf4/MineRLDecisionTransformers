import numpy as np
import torch
import torch.nn as nn


class TrajectoryModel(nn.Module):

    def __init__(self, state_dim, act_dim, max_length=None):
        super().__init__()

        self.state_dim = state_dim
        self.act_dim = act_dim
        self.max_length = max_length

    def forward(self, pov, actions, rewards, returns_to_go,timesteps,state_vector=None, attention_mask=None):
        # "masked" tokens or unspecified inputs can be passed in as None
        return None, None, None

    def get_action(self, states_pov, actions, rewards, returns_to_go, timesteps,states_vector=None, **kwargs):
        # these will come as tensors on the correct device
        return torch.zeros_like(actions[-1])
