import torch
import torch.nn as nn

import enumerateOptions


class PolicyModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, action_dim=None):
        super().__init__()
        if action_dim is None:
            action_dim = enumerateOptions.passInd + 1
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden_dim, action_dim)

    def forward(self, x, action_mask):
        h = self.backbone(x)
        logits = self.head(h)
        mask = action_mask.to(dtype=torch.bool)
        logits = logits.masked_fill(~mask, -1.0e9)
        return logits
