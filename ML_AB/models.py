import torch
import torch.nn as nn

from ML_AB.actions import ACTION_DIM, ACTION_FEAT_DIM
from ML_AB.state import CARD_FEAT_DIM, GLOBAL_FEAT_DIM, HISTORY_FEAT_DIM, HISTORY_LEN


class Big2TransformerNet(nn.Module):
    def __init__(
        self,
        d_model=128,
        nhead=4,
        num_layers=3,
        dropout=0.1,
        action_feat_dim=ACTION_FEAT_DIM,
        max_history_len=HISTORY_LEN,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.max_history_len = int(max_history_len)
        self.card_proj = nn.Linear(CARD_FEAT_DIM, d_model)
        self.history_proj = nn.Linear(HISTORY_FEAT_DIM, d_model)
        self.global_proj = nn.Linear(GLOBAL_FEAT_DIM, d_model)
        self.history_pos = nn.Parameter(torch.zeros(1, self.max_history_len, d_model))
        self.action_proj = nn.Sequential(
            nn.Linear(action_feat_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.value_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.belief_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 3),
        )
        self.logit_scale = nn.Parameter(torch.tensor(1.0))
        self.action_bias = nn.Linear(action_feat_dim, 1)

    def encode(self, card_feats, history_feats, global_feats):
        g = self.global_proj(global_feats).unsqueeze(1)
        c = self.card_proj(card_feats)
        h = self.history_proj(history_feats)
        h = h + self.history_pos[:, : h.shape[1], :]
        tokens = torch.cat([g, c, h], dim=1)
        encoded = self.encoder(tokens)
        return encoded[:, 0], encoded[:, 1:53]

    def forward(self, card_feats, history_feats, global_feats, action_feats, action_mask=None):
        state, card_tokens = self.encode(card_feats, history_feats, global_feats)
        action_emb = self.action_proj(action_feats)
        logits = torch.matmul(action_emb, state.unsqueeze(-1)).squeeze(-1)
        logits = logits / (self.d_model ** 0.5) * self.logit_scale.clamp(0.1, 10.0)
        logits = logits + self.action_bias(action_feats).squeeze(-1)
        if action_mask is not None:
            logits = logits.masked_fill(action_mask <= 0, -1.0e9)
        value = torch.tanh(self.value_head(state)).squeeze(-1)
        belief_logits = self.belief_head(card_tokens)
        return logits, value, belief_logits


def checkpoint_payload(model, config, metrics=None):
    return {
        "format": "ML_AB.Big2TransformerNet.v1",
        "model_state": model.state_dict(),
        "config": dict(config),
        "metrics": dict(metrics or {}),
        "history_len": HISTORY_LEN,
        "action_dim": ACTION_DIM,
        "action_feat_dim": ACTION_FEAT_DIM,
    }
