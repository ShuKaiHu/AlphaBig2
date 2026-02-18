import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import big2Game
from ML_SKHU.policy_value import PolicyValueModel
from ML_SKHU.policy_only.eval_reward import play_game as eval_play_game
from ML_SKHU.features import encode_belief_input, encode_full_info_input


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save", required=True)
    parser.add_argument("--init-from", default=None)
    parser.add_argument("--value-weight", type=float, default=1.0)
    parser.add_argument("--action-weight", type=float, default=0.0)
    parser.add_argument("--entropy-weight", type=float, default=0.01)
    parser.add_argument("--ppo-clip", type=float, default=0.2)
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--eval-ckpt", default=None)
    parser.add_argument("--eval-games", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=0)
    args = parser.parse_args()

    data = np.load(args.data)
    obs = torch.tensor(data["obs"], dtype=torch.float32).to(args.device)
    masks = torch.tensor(data["mask"], dtype=torch.float32).to(args.device)
    actions = torch.tensor(data["actions"], dtype=torch.long).to(args.device)
    old_logprobs = torch.tensor(data["old_logprobs"], dtype=torch.float32).to(args.device)
    rewards = torch.tensor(data["values"], dtype=torch.float32).to(args.device)
    r_mean = rewards.mean()
    r_std = rewards.std().clamp_min(1e-6)
    rewards = (rewards - r_mean) / r_std

    p_input_dim = obs.shape[1]
    full_info = False
    if p_input_dim == encode_full_info_input(big2Game.big2Game(), 1).shape[0]:
        full_info = True
    model = PolicyValueModel(p_input_dim, hidden_dim=256).to(args.device)
    if args.init_from:
        model.load_state_dict(torch.load(args.init_from, map_location=args.device))
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_eval_reward = None
    best_epoch = None
    best_state = None
    no_improve = 0

    for epoch in range(args.epochs):
        idx = torch.randperm(obs.size(0))
        total_loss = 0.0
        total_action_loss = 0.0
        total_value_loss = 0.0
        total_adv = 0.0
        total_adv_pos = 0
        total_adv_neg = 0
        total_adv_abs = 0.0
        total_entropy = 0.0
        total_kl = 0.0
        total_clip_frac = 0.0
        for i in range(0, obs.size(0), args.batch_size):
            batch_idx = idx[i : i + args.batch_size]
            batch_x = obs[batch_idx]
            batch_mask = masks[batch_idx]
            batch_actions = actions[batch_idx]
            batch_rewards = rewards[batch_idx]
            batch_old_logp = old_logprobs[batch_idx]

            logits, values = model(batch_x, batch_mask)
            log_probs = nn.functional.log_softmax(logits, dim=-1)
            selected_log_probs = log_probs[torch.arange(log_probs.size(0)), batch_actions]
            advantages = (batch_rewards.view(-1) - values.view(-1)).detach()
            probs = torch.softmax(logits, dim=-1)
            entropy = -(probs * log_probs).sum(dim=-1).mean()
            if args.ppo_clip > 0:
                ratio = torch.exp(selected_log_probs - batch_old_logp)
                clipped = torch.clamp(ratio, 1.0 - args.ppo_clip, 1.0 + args.ppo_clip)
                action_loss = -(torch.min(ratio * advantages, clipped * advantages)).mean()
                approx_kl = (batch_old_logp - selected_log_probs).mean()
                clip_frac = ((ratio > 1.0 + args.ppo_clip) | (ratio < 1.0 - args.ppo_clip)).float().mean()
            else:
                action_loss = -(selected_log_probs * advantages).mean()
                approx_kl = torch.tensor(0.0, device=logits.device)
                clip_frac = torch.tensor(0.0, device=logits.device)
            value_loss = nn.functional.mse_loss(values.view(-1), batch_rewards.view(-1))
            loss = (
                args.action_weight * action_loss
                + args.value_weight * value_loss
                - args.entropy_weight * entropy
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_x.size(0)
            total_action_loss += action_loss.item() * batch_x.size(0)
            total_value_loss += value_loss.item() * batch_x.size(0)
            total_adv += advantages.sum().item()
            total_adv_pos += int((advantages > 0).sum().item())
            total_adv_neg += int((advantages < 0).sum().item())
            total_adv_abs += advantages.abs().sum().item()
            total_entropy += entropy.item() * batch_x.size(0)
            total_kl += approx_kl.item() * batch_x.size(0)
            total_clip_frac += clip_frac.item() * batch_x.size(0)

        avg_loss = total_loss / obs.size(0)
        avg_action_loss = total_action_loss / obs.size(0)
        avg_value_loss = total_value_loss / obs.size(0)
        avg_adv = total_adv / obs.size(0)
        avg_adv_abs = total_adv_abs / obs.size(0)
        avg_entropy = total_entropy / obs.size(0)
        avg_kl = total_kl / obs.size(0)
        avg_clip_frac = total_clip_frac / obs.size(0)
        adv_pos_ratio = total_adv_pos / max(1, (total_adv_pos + total_adv_neg))
        log_this_epoch = epoch % 10 == 0 or epoch == args.epochs - 1
        eval_this_epoch = (
            args.eval_games > 0
            and args.eval_every > 0
            and (epoch % args.eval_every == 0 or epoch == args.epochs - 1)
        )
        if log_this_epoch or eval_this_epoch:
            line = (
                f"epoch={epoch} loss={avg_loss:.4f} "
                f"action={avg_action_loss:.4f} "
                f"value={avg_value_loss:.4f} "
                f"adv_mean={avg_adv:.4f} "
                f"adv_abs={avg_adv_abs:.4f} "
                f"adv_pos={adv_pos_ratio:.2f}"
            )
            if args.entropy_weight > 0:
                line += f" entropy={avg_entropy:.4f}"
            if args.ppo_clip > 0:
                line += f" approx_kl={avg_kl:.4f} clip_frac={avg_clip_frac:.4f}"
            if eval_this_epoch:
                model.eval()
                eval_rewards = []
                for _ in range(args.eval_games):
                    eval_rewards.append(
                        eval_play_game(model, device=args.device, full_info=full_info)
                    )
                avg_reward = float(np.mean(eval_rewards)) if eval_rewards else 0.0
                line += f" eval_reward={avg_reward:.6f}"
                if best_eval_reward is None or avg_reward > best_eval_reward:
                    best_eval_reward = avg_reward
                    best_epoch = epoch
                    best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1
                model.train()
            print(line)
            if eval_this_epoch and args.early_stop_patience > 0 and no_improve >= args.early_stop_patience:
                print(f"early_stop at epoch={epoch} best_reward={best_eval_reward:.6f}")
                break
        if args.early_stop_patience > 0 and no_improve >= args.early_stop_patience:
            break

    if best_state is not None:
        torch.save(best_state, args.save)
        print(f"saved {args.save} (best eval epoch={best_epoch} reward={best_eval_reward:.6f})")
    else:
        torch.save(model.state_dict(), args.save)
        print(f"saved {args.save}")


if __name__ == "__main__":
    main()
