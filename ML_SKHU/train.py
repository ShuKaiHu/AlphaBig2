import argparse
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from ML_SKHU.belief import BeliefModel
from ML_SKHU.policy_value import PolicyValueModel
from ML_SKHU.features import belief_input_dim
from ML_SKHU.selfplay import run_selfplay_episode
from ML_SKHU.dataset import ReplayBuffer


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


def train_policy_value(model, batch, device="cpu", policy_weight=3.0):
    model.train()
    x, policy_targets, value_targets = zip(*batch)
    x = torch.tensor(np.array(x), dtype=torch.float32, device=device)
    policy_targets = torch.tensor(np.array(policy_targets), dtype=torch.float32, device=device)
    value_targets = torch.tensor(np.array(value_targets), dtype=torch.float32, device=device).unsqueeze(1)
    policy_logits, values = model(x)
    policy_loss = -(policy_targets * nn.functional.log_softmax(policy_logits, dim=-1)).sum(dim=-1).mean()
    value_loss = nn.functional.mse_loss(values, value_targets)
    return policy_weight * policy_loss + value_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--simulations", type=int, default=10)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-hours", type=float, default=0.0)
    parser.add_argument("--assume-yes", action="store_true")
    parser.add_argument("--opponent", type=str, default="selfplay", choices=["selfplay", "heuristic"])
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

    if args.max_hours and not args.assume_yes:
        reply = input(f"Train up to {args.max_hours:.2f} hours? [y/N]: ").strip().lower()
        if reply not in ("y", "yes"):
            return
    deadline = None
    if args.max_hours:
        deadline = time.time() + args.max_hours * 3600.0

    for episode in range(1, args.episodes + 1):
        if deadline is not None and time.time() >= deadline:
            print("Reached max training time. Stopping.")
            break
        start_ts = time.time()
        belief_data, policy_data = run_selfplay_episode(
            belief_model=belief_model,
            policy_value_model=policy_value_model,
            n_simulations=args.simulations,
            temperature=1.0,
            device=device,
            policy_player=1,
            opponent_policy=args.opponent,
        )
        selfplay_time = time.time() - start_ts
        for item in belief_data:
            buffer.add_belief(*item)
        for item in policy_data:
            buffer.add_policy(*item)

        b_loss = None
        if len(buffer.belief) > 0:
            batch = buffer.sample_belief(args.batch_size)
            loss = train_belief(belief_model, batch, device=device)
            b_optim.zero_grad()
            loss.backward()
            b_optim.step()
            b_loss = float(loss.item())

        p_loss = None
        if len(buffer.policy) > 0:
            batch = buffer.sample_policy(args.batch_size)
            loss = train_policy_value(policy_value_model, batch, device=device)
            p_optim.zero_grad()
            loss.backward()
            p_optim.step()
            p_loss = float(loss.item())

        if episode == 1 or episode % 5 == 0:
            msg = (
                f"episode={episode}/{args.episodes} "
                f"selfplay={selfplay_time:.1f}s "
                f"belief_loss={b_loss if b_loss is not None else 'n/a'} "
                f"policy_loss={p_loss if p_loss is not None else 'n/a'} "
                f"buffer(belief,policy)={len(buffer.belief)},{len(buffer.policy)}"
            )
            print(msg)

    torch.save(belief_model.state_dict(), "ml2/models/belief.pt")
    torch.save(policy_value_model.state_dict(), "ml2/models/policy_value.pt")


if __name__ == "__main__":
    main()
