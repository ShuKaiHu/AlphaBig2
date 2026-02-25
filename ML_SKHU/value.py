import torch
import torch.nn as nn


class ValueModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        # Value target is tanh-scaled reward in [-1, 1], so clamp model output
        # to the same range to avoid overconfident values destabilizing MCTS.
        return torch.tanh(self.net(x)).squeeze(-1)
