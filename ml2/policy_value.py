import torch
import torch.nn as nn


class PolicyValueModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, action_dim=1695):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: [batch, input_dim]
        h = self.shared(x)
        policy_logits = self.policy_head(h)
        value = self.value_head(h)
        return policy_logits, value
