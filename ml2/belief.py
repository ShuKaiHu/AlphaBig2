import torch
import torch.nn as nn


class BeliefModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 52 * 3),
        )

    def forward(self, x):
        # x: [batch, input_dim]
        logits = self.net(x)
        return logits.view(-1, 52, 3)
