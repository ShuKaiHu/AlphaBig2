import argparse
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import big2Game

from ML_SKHU.belief import BeliefModel
from ML_SKHU.policy_value import PolicyValueModel
from ML_SKHU.features import belief_input_dim, encode_full_info_with_belief
from ML_SKHU.selfplay import run_selfplay_episode
from ML_SKHU.eval import play_game as eval_play_game
from ML_SKHU.dataset import ReplayBuffer


def train_belief(model, batch, device="cpu"):
    model.train()
    if len(batch[0]) == 3:
        x, targets, masks = zip(*batch)
        weights = [1.0] * len(batch)
    else:
        x, targets, masks, weights = zip(*batch)
    x = torch.tensor(np.array(x), dtype=torch.float32, device=device)
    targets = torch.tensor(np.array(targets), dtype=torch.long, device=device)
    masks = torch.tensor(np.array(masks), dtype=torch.float32, device=device)
    weights = torch.tensor(np.array(weights), dtype=torch.float32, device=device)
    logits = model(x)
    loss = nn.functional.cross_entropy(
        logits.view(-1, 3),
        targets.view(-1),
        reduction="none",
        ignore_index=-1,
    )
    sample_weights = weights.view(-1, 1).expand_as(masks)
    weighted_masks = masks * sample_weights
    loss = (loss * weighted_masks.view(-1)).sum() / (weighted_masks.sum() + 1e-6)
    probs = torch.softmax(logits, dim=-1)
    uniform = torch.full_like(probs, 1.0 / 3.0)
    kl = (probs * (torch.log(probs + 1e-8) - torch.log(uniform))).sum(dim=-1)
    kl = (kl * weighted_masks).sum() / (weighted_masks.sum() + 1e-6)
    loss = loss + 0.1 * kl
    return loss


def train_policy_value(model, batch, device="cpu", policy_weight=3.0):
    model.train()
    x, policy_targets, value_targets, action_masks = zip(*batch)
    x = torch.tensor(np.array(x), dtype=torch.float32, device=device)
    policy_targets = torch.tensor(np.array(policy_targets), dtype=torch.float32, device=device)
    value_targets = torch.tensor(np.array(value_targets), dtype=torch.float32, device=device)
    action_masks = torch.tensor(np.array(action_masks), dtype=torch.float32, device=device)
    policy_logits, values = model(x, action_masks)
    policy_loss = -(policy_targets * nn.functional.log_softmax(policy_logits, dim=-1)).sum(dim=-1).mean()
    value_loss = nn.functional.mse_loss(values.view(-1), value_targets.view(-1))
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
    parser.add_argument("--opponent", type=str, default="random", choices=["selfplay", "heuristic", "random"])
    parser.add_argument("--policy-weight", type=float, default=1.0)
    parser.add_argument("--updates-per-episode", type=int, default=1)
    parser.add_argument("--train-belief", action="store_true")
    parser.add_argument("--train-policy", action="store_true")
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--eval-games", type=int, default=10)
    parser.add_argument("--belief-margin", type=float, default=0.5)
    parser.add_argument("--policy-only-ckpt", type=str, default="")
    parser.add_argument("--policy-only-player", type=int, default=1)
    parser.add_argument("--belief-ckpt", type=str, default="")
    parser.add_argument("--policy-ckpt", type=str, default="")
    parser.add_argument("--belief-min-turn", type=int, default=0)
    parser.add_argument("--belief-decay", type=float, default=0.5)
    args = parser.parse_args()

    device = args.device
    b_input_dim = belief_input_dim()
    dummy = big2Game.big2Game()
    dummy_belief = np.zeros((52, 3), dtype=np.float32)
    p_input_dim = int(encode_full_info_with_belief(dummy, 1, dummy_belief).shape[0])
    belief_model = BeliefModel(b_input_dim, num_layers=3).to(device)
    policy_value_model = PolicyValueModel(p_input_dim).to(device)
    policy_only_model = None

    model_dir = f"ML_SKHU/models/{args.opponent}"
    os.makedirs(model_dir, exist_ok=True)
    if args.belief_ckpt:
        try:
            belief_model.load_state_dict(torch.load(args.belief_ckpt, map_location=device))
        except Exception as e:
            print(f"[warn] belief ckpt load failed: {e}")
    elif args.resume:
        b_path = f"{model_dir}/belief.pt"
        p_path = f"{model_dir}/policy_value.pt"
        if os.path.exists(b_path):
            belief_model.load_state_dict(torch.load(b_path, map_location=device))
        if os.path.exists(p_path):
            policy_value_model.load_state_dict(torch.load(p_path, map_location=device))
    if args.policy_ckpt:
        policy_value_model.load_state_dict(torch.load(args.policy_ckpt, map_location=device))

    if args.policy_only_ckpt:
        try:
            policy_only_model = PolicyValueModel(b_input_dim, hidden_dim=256).to(device)
            policy_only_model.load_state_dict(torch.load(args.policy_only_ckpt, map_location=device))
            policy_only_model.eval()
        except Exception as e:
            print(f"[warn] policy-only ckpt load failed: {e}")
            policy_only_model = None

    buffer = ReplayBuffer(capacity=50000)
    b_optim = optim.Adam(belief_model.parameters(), lr=1e-3)
    p_optim = optim.Adam(policy_value_model.parameters(), lr=1e-3)

    if args.max_hours and not args.assume_yes:
        reply = input(f"Train up to {args.max_hours:.2f} hours? [y/N]: ").strip().lower()
        if reply not in ("y", "yes"):
            return
    deadline = None
    if args.max_hours:
        deadline = time.time() + args.max_hours * 3600.0

    train_belief_flag = args.train_belief or not args.train_policy
    train_policy_flag = args.train_policy or not args.train_belief

    for episode in range(1, args.episodes + 1):
        if deadline is not None and time.time() >= deadline:
            print("Reached max training time. Stopping.")
            break
        start_ts = time.time()
        belief_data, policy_data = run_selfplay_episode(
            belief_model=belief_model,
            policy_value_model=policy_value_model,
            policy_only_model=policy_only_model,
            n_simulations=args.simulations,
            temperature=1.0,
            device=device,
            policy_player=args.policy_only_player,
            opponent_policy=args.opponent,
            belief_min_turn=args.belief_min_turn,
            belief_decay=args.belief_decay,
        )
        selfplay_time = time.time() - start_ts
        for item in belief_data:
            buffer.add_belief(*item)
        for item in policy_data:
            buffer.add_policy(*item)

        b_loss = None
        p_loss = None
        for _ in range(max(1, args.updates_per_episode)):
            if train_belief_flag and len(buffer.belief) > 0:
                batch = buffer.sample_belief(args.batch_size)
                loss = train_belief(belief_model, batch, device=device)
                b_optim.zero_grad()
                loss.backward()
                b_optim.step()
                b_loss = float(loss.item())

            if train_policy_flag and len(buffer.policy) > 0:
                batch = buffer.sample_policy(args.batch_size)
                loss = train_policy_value(
                    policy_value_model,
                    batch,
                    device=device,
                    policy_weight=args.policy_weight,
                )
                p_optim.zero_grad()
                loss.backward()
                p_optim.step()
                p_loss = float(loss.item())

        do_log = args.log_every > 0 and (episode % args.log_every == 0)
        do_eval = args.eval_every > 0 and (episode == args.episodes or episode % args.eval_every == 0)
        if do_log or do_eval:
            b_disp = f"{b_loss:.2f}" if b_loss is not None else "n/a"
            p_disp = f"{p_loss:.2f}" if p_loss is not None else "n/a"

            eval_rewards = []
            belief_accs = []
            belief_covs = []
            known_counts = []
            valid_counts = []
            phase_stats = None
            if do_eval:
                for _ in range(args.eval_games):
                    wins, rewards, accs, covs, kcounts, vcounts, phases = eval_play_game(
                        belief_model,
                        policy_value_model,
                        policy_only_model,
                        args.simulations,
                        args.opponent,
                        device=device,
                        margin=args.belief_margin,
                    )
                    eval_rewards.append(rewards[0])
                    belief_accs.extend(accs)
                    belief_covs.extend(covs)
                    known_counts.extend(kcounts)
                    valid_counts.extend(vcounts)
                    # aggregate phase stats
                    if phase_stats is None:
                        phase_stats = {
                            "early": {"acc_sum": 0.0, "acc_n": 0, "cov_sum": 0.0, "cov_n": 0, "known": 0, "valid": 0},
                            "mid": {"acc_sum": 0.0, "acc_n": 0, "cov_sum": 0.0, "cov_n": 0, "known": 0, "valid": 0},
                            "late": {"acc_sum": 0.0, "acc_n": 0, "cov_sum": 0.0, "cov_n": 0, "known": 0, "valid": 0},
                        }
                    for key in phase_stats:
                        phase_stats[key]["acc_sum"] += phases[key]["acc_sum"]
                        phase_stats[key]["acc_n"] += phases[key]["acc_n"]
                        phase_stats[key]["cov_sum"] += phases[key]["cov_sum"]
                        phase_stats[key]["cov_n"] += phases[key]["cov_n"]
                        phase_stats[key]["known"] += phases[key]["known"]
                        phase_stats[key]["valid"] += phases[key]["valid"]

            avg_eval_reward = float(np.mean(eval_rewards)) if eval_rewards else 0.0
            avg_belief = float(np.mean(belief_accs)) if belief_accs else 0.0
            avg_cov = float(np.mean(belief_covs)) if belief_covs else 0.0
            total_known = int(np.sum(known_counts)) if known_counts else 0
            total_valid = int(np.sum(valid_counts)) if valid_counts else 0
            if phase_stats:
                def _phase_avg(d):
                    acc = (d["acc_sum"] / d["acc_n"]) if d["acc_n"] > 0 else 0.0
                    cov = (d["cov_sum"] / d["cov_n"]) if d["cov_n"] > 0 else 0.0
                    return acc, cov
                e_acc, e_cov = _phase_avg(phase_stats["early"])
                m_acc, m_cov = _phase_avg(phase_stats["mid"])
                l_acc, l_cov = _phase_avg(phase_stats["late"])
            else:
                e_acc = e_cov = m_acc = m_cov = l_acc = l_cov = 0.0

            line = f"episode={episode}/{args.episodes} selfplay={selfplay_time:.1f}s "
            if train_belief_flag:
                line += f"belief_loss={b_disp} "
            if train_policy_flag:
                line += f"policy_loss={p_disp} "
            line += (
                f"eval_reward={avg_eval_reward:.2f} "
                f"acc_known={avg_belief:.2f} "
                f"known_rate={avg_cov:.2f} "
                f"known_count={total_known}/{total_valid} "
                f"acc_e/m/l={e_acc:.2f}/{m_acc:.2f}/{l_acc:.2f} "
                f"kr_e/m/l={e_cov:.2f}/{m_cov:.2f}/{l_cov:.2f} "
            )
            if train_belief_flag and train_policy_flag:
                line += f"buffer(belief,policy)={len(buffer.belief)},{len(buffer.policy)}"
            if do_log:
                print(line)
            if do_eval and not do_log:
                print(line)

    torch.save(belief_model.state_dict(), f"{model_dir}/belief.pt")
    torch.save(policy_value_model.state_dict(), f"{model_dir}/policy_value.pt")


if __name__ == "__main__":
    main()
