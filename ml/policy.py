import torch
from torch import nn


class PolicyNet(nn.Module):
    def __init__(self, obs_dim=412, act_dim=1695, hidden_sizes=(512, 256)):
        super().__init__()
        h1, h2 = hidden_sizes
        self.net = nn.Sequential(
            nn.Linear(obs_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, act_dim),
        )

    def forward(self, obs, action_mask):
        logits = self.net(obs)
        mask = action_mask.to(dtype=torch.bool)
        logits = logits.masked_fill(~mask, -1.0e9)
        return logits


class ActorCriticNet(nn.Module):
    def __init__(self, obs_dim=412, act_dim=1695, hidden_sizes=(512, 256)):
        super().__init__()
        h1, h2 = hidden_sizes
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(h2, act_dim)
        self.value_head = nn.Linear(h2, 1)

    def forward(self, obs, action_mask):
        x = self.trunk(obs)
        logits = self.policy_head(x)
        mask = action_mask.to(dtype=torch.bool)
        logits = logits.masked_fill(~mask, -1.0e9)
        value = self.value_head(x).squeeze(-1)
        return logits, value
