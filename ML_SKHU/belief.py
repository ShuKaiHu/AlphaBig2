import torch
import torch.nn as nn


class BeliefModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, num_layers=3):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(max(1, num_layers) - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, 52 * 3))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: [batch, input_dim]
        logits = self.net(x)
        return logits.view(-1, 52, 3)
