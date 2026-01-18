import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from ml2.belief import BeliefModel
from ml2.policy_value import PolicyValueModel
from ml2.features import belief_input_dim
from ml2.selfplay import run_selfplay_episode
from ml2.dataset import ReplayBuffer


def train_belief(model, batch, device="cpu"):
    model.train()
    x, targets, masks = zip(*batch)
    x = torch.tensor(np.array(x), dtype=torch.float32, device=device)
    targets = torch.tensor(np.array(targets), dtype=torch.long, device=device)
    masks = torch.tensor(np.array(masks), dtype=torch.float32, device=device)
    logits = model(x)
    loss = nn.functional.cross_entropy(
        logits.view(-1, 3),
        targets.view(-1),
        reduction="none",
        ignore_index=-1,
    )
    loss = (loss * masks.view(-1)).sum() / (masks.sum() + 1e-6)
    return loss


def train_policy_value(model, batch, device="cpu"):
    model.train()
    x, policy_targets, value_targets = zip(*batch)
    x = torch.tensor(np.array(x), dtype=torch.float32, device=device)
    policy_targets = torch.tensor(np.array(policy_targets), dtype=torch.float32, device=device)
    value_targets = torch.tensor(np.array(value_targets), dtype=torch.float32, device=device).unsqueeze(1)
    policy_logits, values = model(x)
    policy_loss = -(policy_targets * nn.functional.log_softmax(policy_logits, dim=-1)).sum(dim=-1).mean()
    value_loss = nn.functional.mse_loss(values, value_targets)
    return policy_loss + value_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--simulations", type=int, default=10)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    device = args.device
    b_input_dim = belief_input_dim()
    p_input_dim = b_input_dim + 52 * 3
    belief_model = BeliefModel(b_input_dim).to(device)
    policy_value_model = PolicyValueModel(p_input_dim).to(device)

    os.makedirs("ml2/models", exist_ok=True)
    if args.resume:
        b_path = "ml2/models/belief.pt"
        p_path = "ml2/models/policy_value.pt"
        if os.path.exists(b_path):
            belief_model.load_state_dict(torch.load(b_path, map_location=device))
        if os.path.exists(p_path):
            policy_value_model.load_state_dict(torch.load(p_path, map_location=device))

    buffer = ReplayBuffer(capacity=20000)
    b_optim = optim.Adam(belief_model.parameters(), lr=1e-3)
    p_optim = optim.Adam(policy_value_model.parameters(), lr=1e-3)

    for _ in range(args.episodes):
        belief_data, policy_data = run_selfplay_episode(
            belief_model=belief_model,
            policy_value_model=policy_value_model,
            n_simulations=args.simulations,
            temperature=1.0,
            device=device,
        )
        for item in belief_data:
            buffer.add_belief(*item)
        for item in policy_data:
            buffer.add_policy(*item)

        if len(buffer.belief) > 0:
            batch = buffer.sample_belief(args.batch_size)
            loss = train_belief(belief_model, batch, device=device)
            b_optim.zero_grad()
            loss.backward()
            b_optim.step()

        if len(buffer.policy) > 0:
            batch = buffer.sample_policy(args.batch_size)
            loss = train_policy_value(policy_value_model, batch, device=device)
            p_optim.zero_grad()
            loss.backward()
            p_optim.step()

    torch.save(belief_model.state_dict(), "ml2/models/belief.pt")
    torch.save(policy_value_model.state_dict(), "ml2/models/policy_value.pt")


if __name__ == "__main__":
    main()
