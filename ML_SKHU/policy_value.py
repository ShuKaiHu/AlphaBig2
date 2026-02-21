import torch
import torch.nn as nn
import enumerateOptions


class PolicyValueModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, action_dim=None):
        super().__init__()
        if action_dim is None:
            action_dim = enumerateOptions.passInd + 1
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x, action_mask):
        # x: [batch, input_dim]
        h = self.shared(x)
        policy_logits = self.policy_head(h)
        value = self.value_head(h).squeeze(-1)

        unknown_mask = x[..., -52:]
        unknown_score = unknown_mask.mean(dim=-1, keepdim=True)
        scaling = 1.0 - torch.clamp(unknown_score, 0.0, 1.0)
        policy_logits = policy_logits * scaling

        mask = action_mask.to(dtype=torch.bool)
        policy_logits = policy_logits.masked_fill(~mask, -1.0e9)
        return policy_logits, value
