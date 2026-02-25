import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import big2Game

from ML_SKHU.belief import BeliefModel
from ML_SKHU.policy import PolicyModel
from ML_SKHU.value import ValueModel
from ML_SKHU.features import belief_input_dim, encode_policy_input
from ML_SKHU.selfplay import run_selfplay_episode
from ML_SKHU.dataset import ReplayBuffer


def _lerp(start, end, t):
    return float(start + (end - start) * t)


def train_belief_step(model, batch, device="cpu"):
    x, targets, masks, weights = zip(*batch)
    x = torch.tensor(np.array(x), dtype=torch.float32, device=device)
    targets = torch.tensor(np.array(targets), dtype=torch.long, device=device)
    masks = torch.tensor(np.array(masks), dtype=torch.float32, device=device)
    weights = torch.tensor(np.array(weights), dtype=torch.float32, device=device)

    logits = model(x)  # [B,52,3]
    loss = nn.functional.cross_entropy(
        logits.view(-1, 3),
        targets.view(-1),
        reduction="none",
        ignore_index=-1,
    )
    wm = masks * weights.view(-1, 1)
    loss = (loss * wm.view(-1)).sum() / (wm.sum() + 1e-6)
    return loss


def train_policy_value_step(policy_model, value_model, batch, device="cpu"):
    x, policy_targets, value_targets, action_masks = zip(*batch)
    x = torch.tensor(np.array(x), dtype=torch.float32, device=device)
    policy_targets = torch.tensor(np.array(policy_targets), dtype=torch.float32, device=device)
    value_targets = torch.tensor(np.array(value_targets), dtype=torch.float32, device=device)
    action_masks = torch.tensor(np.array(action_masks), dtype=torch.float32, device=device)

    logits = policy_model(x, action_masks)
    values = value_model(x)

    policy_loss = -(policy_targets * nn.functional.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
    value_loss = nn.functional.mse_loss(values.view(-1), value_targets.view(-1))
    return policy_loss, value_loss


def belief_accuracy(model, batch, device="cpu"):
    x, targets, masks, _weights = zip(*batch)
    x = torch.tensor(np.array(x), dtype=torch.float32, device=device)
    targets = torch.tensor(np.array(targets), dtype=torch.long, device=device)
    masks = torch.tensor(np.array(masks), dtype=torch.float32, device=device)
    with torch.no_grad():
        logits = model(x)  # [B,52,3]
        pred = torch.argmax(logits, dim=-1)  # [B,52]
    valid = (masks > 0) & (targets >= 0)
    if valid.sum().item() == 0:
        return 0.0
    acc = (pred[valid] == targets[valid]).float().mean().item()
    return float(acc)


def policy_top1_accuracy(policy_model, batch, device="cpu"):
    x, policy_targets, _value_targets, action_masks = zip(*batch)
    x = torch.tensor(np.array(x), dtype=torch.float32, device=device)
    policy_targets = torch.tensor(np.array(policy_targets), dtype=torch.float32, device=device)
    action_masks = torch.tensor(np.array(action_masks), dtype=torch.float32, device=device)
    with torch.no_grad():
        logits = policy_model(x, action_masks)
        pred_action = torch.argmax(logits, dim=-1)
        target_action = torch.argmax(policy_targets, dim=-1)
    acc = (pred_action == target_action).float().mean().item()
    return float(acc)


def policy_target_expected_value_gap(policy_model, batch, device="cpu"):
    """
    Proxy metric using available MCTS policy targets only:
    gap = max(target_prob) - target_prob[predicted_action]
    Lower is better. 0 means policy picked a target-argmax action.
    """
    x, policy_targets, _value_targets, action_masks = zip(*batch)
    x = torch.tensor(np.array(x), dtype=torch.float32, device=device)
    policy_targets = torch.tensor(np.array(policy_targets), dtype=torch.float32, device=device)
    action_masks = torch.tensor(np.array(action_masks), dtype=torch.float32, device=device)
    with torch.no_grad():
        logits = policy_model(x, action_masks)
        pred_action = torch.argmax(logits, dim=-1)  # [B]
        target_max = torch.max(policy_targets, dim=-1).values  # [B]
        picked_prob = policy_targets.gather(1, pred_action.view(-1, 1)).squeeze(1)  # [B]
        gap = target_max - picked_prob
    return float(gap.mean().item())


def value_mae(value_model, batch, device="cpu"):
    x, _policy_targets, value_targets, _action_masks = zip(*batch)
    x = torch.tensor(np.array(x), dtype=torch.float32, device=device)
    value_targets = torch.tensor(np.array(value_targets), dtype=torch.float32, device=device)
    with torch.no_grad():
        pred = value_model(x).view(-1)
    mae = torch.mean(torch.abs(pred - value_targets.view(-1))).item()
    return float(mae)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--simulations", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--updates-per-episode", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--opponent", choices=["heuristic", "random"], default="heuristic")
    parser.add_argument("--belief-mode", choices=["model", "oracle", "uniform"], default="model")
    parser.add_argument("--policy-player", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--policy-weight", type=float, default=1.0)
    parser.add_argument("--value-weight", type=float, default=1.0)
    parser.add_argument("--train-belief", action="store_true")
    parser.add_argument("--value-scale", type=float, default=15.0)
    parser.add_argument("--val-size", type=int, default=256)
    parser.add_argument("--selfplay-temp-start", type=float, default=1.0)
    parser.add_argument("--selfplay-temp-end", type=float, default=0.5)
    parser.add_argument("--target-temp-start", type=float, default=1.0)
    parser.add_argument("--target-temp-end", type=float, default=0.5)
    parser.add_argument("--dirichlet-alpha", type=float, default=0.3)
    parser.add_argument("--dirichlet-frac-start", type=float, default=0.25)
    parser.add_argument("--dirichlet-frac-end", type=float, default=0.05)
    parser.add_argument("--belief-ckpt", type=str, default="")
    parser.add_argument("--policy-ckpt", type=str, default="")
    parser.add_argument("--value-ckpt", type=str, default="")
    parser.add_argument("--save-dir", type=str, default="ML_SKHU/models/new")
    args = parser.parse_args()

    device = args.device
    os.makedirs(args.save_dir, exist_ok=True)

    b_dim = belief_input_dim()
    dummy_game = big2Game.big2Game()
    dummy_belief = np.full((52, 3), 1.0 / 3.0, dtype=np.float32)
    p_dim = int(encode_policy_input(dummy_game, 1, dummy_belief).shape[0])

    belief_model = BeliefModel(b_dim, hidden_dim=512, num_layers=3).to(device)
    policy_model = PolicyModel(p_dim, hidden_dim=512).to(device)
    value_model = ValueModel(p_dim, hidden_dim=512).to(device)

    if args.belief_ckpt:
        belief_model.load_state_dict(torch.load(args.belief_ckpt, map_location=device))
    if args.policy_ckpt:
        policy_model.load_state_dict(torch.load(args.policy_ckpt, map_location=device))
    if args.value_ckpt:
        value_model.load_state_dict(torch.load(args.value_ckpt, map_location=device))

    buffer = ReplayBuffer(capacity=50000)
    b_optim = optim.Adam(belief_model.parameters(), lr=1e-3)
    p_optim = optim.Adam(policy_model.parameters(), lr=1e-3)
    v_optim = optim.Adam(value_model.parameters(), lr=1e-3)
    belief_val_set = None
    policy_val_set = None

    for ep in range(1, args.episodes + 1):
        if args.episodes <= 1:
            progress = 1.0
        else:
            progress = float(ep - 1) / float(args.episodes - 1)
        selfplay_temp = _lerp(args.selfplay_temp_start, args.selfplay_temp_end, progress)
        target_temp = _lerp(args.target_temp_start, args.target_temp_end, progress)
        dirichlet_frac = _lerp(args.dirichlet_frac_start, args.dirichlet_frac_end, progress)
        add_root_noise = dirichlet_frac > 1e-8

        belief_model.eval()
        policy_model.eval()
        value_model.eval()
        belief_data, policy_data = run_selfplay_episode(
            belief_model=belief_model,
            policy_model=policy_model,
            value_model=value_model,
            n_simulations=args.simulations,
            temperature=selfplay_temp,
            device=device,
            policy_player=args.policy_player,
            opponent_policy=args.opponent,
            value_scale=args.value_scale,
            belief_mode=args.belief_mode,
            target_temperature=target_temp,
            dirichlet_alpha=args.dirichlet_alpha,
            dirichlet_frac=dirichlet_frac,
            add_root_noise=add_root_noise,
        )
        for item in belief_data:
            buffer.add_belief(*item)
        for item in policy_data:
            buffer.add_policy(*item)

        if belief_val_set is None and len(buffer.belief) >= args.val_size:
            belief_val_set = list(buffer.sample_belief(args.val_size))
        if policy_val_set is None and len(buffer.policy) >= args.val_size:
            policy_val_set = list(buffer.sample_policy(args.val_size))

        b_loss_val = 0.0
        p_loss_val = 0.0
        v_loss_val = 0.0

        for _ in range(args.updates_per_episode):
            if len(buffer.belief) > 0:
                batch = buffer.sample_belief(args.batch_size)
                if args.train_belief:
                    belief_model.train()
                    b_loss = train_belief_step(belief_model, batch, device=device)
                    b_optim.zero_grad()
                    b_loss.backward()
                    b_optim.step()
                else:
                    belief_model.eval()
                    with torch.no_grad():
                        b_loss = train_belief_step(belief_model, batch, device=device)
                b_loss_val = float(b_loss.item())

            if len(buffer.policy) > 0:
                batch = buffer.sample_policy(args.batch_size)
                policy_model.train()
                value_model.train()
                p_loss, v_loss = train_policy_value_step(
                    policy_model, value_model, batch, device=device
                )
                total = args.policy_weight * p_loss + args.value_weight * v_loss
                p_optim.zero_grad()
                v_optim.zero_grad()
                total.backward()
                p_optim.step()
                v_optim.step()
                p_loss_val = float(p_loss.item())
                v_loss_val = float(v_loss.item())

        if ep % args.log_every == 0 or ep == 1:
            b_len, p_len = buffer.__len__()
            b_acc = None
            p_acc = None
            p_gap = None
            v_mae = None
            if belief_val_set is not None:
                belief_model.eval()
                b_acc = belief_accuracy(belief_model, belief_val_set, device=device)
            if policy_val_set is not None:
                policy_model.eval()
                value_model.eval()
                p_acc = policy_top1_accuracy(policy_model, policy_val_set, device=device)
                p_gap = policy_target_expected_value_gap(
                    policy_model, policy_val_set, device=device
                )
                v_mae = value_mae(value_model, policy_val_set, device=device)
            b_acc_str = f"{b_acc:.4f}" if b_acc is not None else "n/a"
            p_acc_str = f"{p_acc:.4f}" if p_acc is not None else "n/a"
            p_gap_str = f"{p_gap:.4f}" if p_gap is not None else "n/a"
            v_mae_str = f"{v_mae:.4f}" if v_mae is not None else "n/a"
            print(
                f"episode={ep}/{args.episodes} "
                f"belief_loss={b_loss_val:.4f} policy_loss={p_loss_val:.4f} value_loss={v_loss_val:.4f} "
                f"kpi(b_acc,p_top1,target_expected_value_gap,v_mae)="
                f"{b_acc_str},{p_acc_str},{p_gap_str},{v_mae_str} "
                f"buffer(belief,policy)={b_len},{p_len} "
                f"belief_mode={args.belief_mode} "
                f"mcts(temp,target_t,noise)={selfplay_temp:.3f},{target_temp:.3f},{dirichlet_frac:.3f}"
            )

    torch.save(belief_model.state_dict(), os.path.join(args.save_dir, "belief.pt"))
    torch.save(policy_model.state_dict(), os.path.join(args.save_dir, "policy.pt"))
    torch.save(value_model.state_dict(), os.path.join(args.save_dir, "value.pt"))
    print(f"saved models to {args.save_dir}")


if __name__ == "__main__":
    main()
