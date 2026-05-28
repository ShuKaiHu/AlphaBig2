import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from engine.features import STATIC_DIM, GRU_HIDDEN, HIST_STEP_DIM, HISTORY_LEN
import enumerateOptions

ACTION_SIZE = enumerateOptions.passInd + 1  # 14739


class ResBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )

    def forward(self, x):
        return F.relu(x + self.net(x))


class Big2Net(nn.Module):
    """
    AlphaZero-style network for Big 2.

    Architecture:
        GRU(history_seq) → context vector
        concat(static_feat, context) → input_proj → 4 × ResBlock → trunk
        trunk → policy head → logits (ACTION_SIZE,)
        trunk → value head  → scalar in [-1, 1]

    Uses LayerNorm instead of BatchNorm so it works correctly
    with batch_size=1 during MCTS inference.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        n_res_blocks: int = 4,
        gru_hidden: int = GRU_HIDDEN,
        action_size: int = ACTION_SIZE,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=HIST_STEP_DIM,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
        )
        input_dim = STATIC_DIM + gru_hidden
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.res_blocks = nn.ModuleList(
            [ResBlock(hidden_dim) for _ in range(n_res_blocks)]
        )
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_size),
        )
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh(),
        )
        # Belief auxiliary head: predict which cards each of 3 opponents holds
        # Output: 52 * 3 = 156 logits (BCEWithLogits), only used during training
        self.belief_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 52 * 3),
        )

    def forward(self, static_feat, history_seq, valid_mask=None):
        """
        Args:
            static_feat:  (B, STATIC_DIM)   float32
            history_seq:  (B, HISTORY_LEN, HIST_STEP_DIM) float32
            valid_mask:   (B, ACTION_SIZE)  float32, 1=valid, 0=invalid (optional)
        Returns:
            policy_logits: (B, ACTION_SIZE)  — invalid actions masked to -1e9
            value:         (B, 1)            — tanh output ∈ [-1, 1]
            belief_logits: (B, 156)          — raw logits for 3 opponents × 52 cards
        """
        _, h = self.gru(history_seq)          # h: (1, B, gru_hidden)
        context = h.squeeze(0)                # (B, gru_hidden)
        x = torch.cat([static_feat, context], dim=-1)
        x = self.input_proj(x)
        for block in self.res_blocks:
            x = block(x)
        logits = self.policy_head(x)
        if valid_mask is not None:
            logits = logits + (1.0 - valid_mask) * (-1e9)
        value = self.value_head(x)
        belief_logits = self.belief_head(x)
        return logits, value, belief_logits

    @torch.no_grad()
    def predict(self, static_feat, history_seq, valid_mask=None):
        """
        Single-sample inference. Inputs are numpy arrays.
        Returns:
            probs: np.ndarray (ACTION_SIZE,)  softmax probabilities (valid actions only)
            value: float
        """
        self.eval()
        sf = torch.FloatTensor(static_feat).unsqueeze(0)
        hs = torch.FloatTensor(history_seq).unsqueeze(0)
        vm = torch.FloatTensor(valid_mask).unsqueeze(0) if valid_mask is not None else None
        logits, val, _ = self.forward(sf, hs, vm)
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        return probs, float(val.squeeze().item())
