import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import big2Game
import enumerateOptions

from ML_SKHU.value import ValueModel
from ML_SKHU.features import encode_policy_input


def _heuristic_action(game):
    avail = game.returnAvailableActions()
    valid = np.flatnonzero(avail == 1)
    if valid.size == 0:
        return enumerateOptions.passInd
    non_pass = valid[valid != enumerateOptions.passInd]
    if non_pass.size > 0:
        return int(np.min(non_pass))
    return int(enumerateOptions.passInd)


def _apply_action(game, action):
    if action == enumerateOptions.passInd:
        game.updateGame(-1)
    else:
        opt, n = enumerateOptions.getOptionNC(action)
        game.updateGame(opt, n)


def _oracle_belief_probs(game):
    # encode_policy_input expects [52,3] for P2/P3/P4 in absolute player ids
    probs = np.zeros((52, 3), dtype=np.float32)
    for abs_player, ch in [(2, 0), (3, 1), (4, 2)]:
        for cid in game.currentHands[abs_player]:
            probs[int(cid) - 1, ch] = 1.0
    return probs


def collect_episode_samples(value_scale=15.0):
    game = big2Game.big2Game()
    samples = []  # (x, player_idx, total_remaining, max_hand, turn_idx)
    turn_idx = 0

    while not game.gameOver:
        p = game.playersGo  # 1..4
        rem = [len(game.currentHands[i]) for i in [1, 2, 3, 4]]
        total_remaining = int(sum(rem))
        max_hand = int(max(rem))
        x = encode_policy_input(game, p, _oracle_belief_probs(game))
        samples.append((x, p - 1, total_remaining, max_hand, turn_idx))
        a = _heuristic_action(game)
        _apply_action(game, a)
        turn_idx += 1

    rewards = np.array(game.rewards, dtype=np.float32)  # len 4
    total_turns = turn_idx
    out = []
    for x, p_idx, total_remaining, max_hand, t_idx in samples:
        target = np.tanh(float(rewards[p_idx]) / float(value_scale))
        steps_to_end = int(total_turns - t_idx)
        out.append((x, np.float32(target), total_remaining, max_hand, steps_to_end))
    return out


def sample_batch(data, batch_size):
    batch = random.sample(data, min(batch_size, len(data)))
    xs = [b[0] for b in batch]
    ys = [b[1] for b in batch]
    x = torch.tensor(np.array(xs), dtype=torch.float32)
    y = torch.tensor(np.array(ys), dtype=torch.float32)
    return x, y


def _is_endgame_sample(sample, args_or_cfg):
    # sample = (x, y, total_remaining, max_hand, steps_to_end)
    _, _, total_remaining, max_hand, steps_to_end = sample
    if steps_to_end > args_or_cfg["endgame_steps_threshold"]:
        return False
    if args_or_cfg["endgame_total_threshold"] > 0 and total_remaining > args_or_cfg["endgame_total_threshold"]:
        return False
    if args_or_cfg["endgame_max_hand_threshold"] > 0 and max_hand > args_or_cfg["endgame_max_hand_threshold"]:
        return False
    return True


def _step_bucket_oversample_factor(sample, cfg):
    # sample = (x, y, total_remaining, max_hand, steps_to_end)
    steps_to_end = int(sample[4])
    if steps_to_end <= cfg["w_le8_steps"]:
        return int(cfg["w_le8"])
    if steps_to_end <= cfg["w_le16_steps"]:
        return int(cfg["w_le16"])
    if steps_to_end <= cfg["w_le24_steps"]:
        return int(cfg["w_le24"])
    return int(cfg["w_gt24"])


def eval_value(model, data, device="cpu", batch_size=256, end_cfg=None):
    if len(data) == 0:
        return None, None, None, None
    if end_cfg is None:
        end_cfg = {
            "endgame_steps_threshold": 8,
            "endgame_total_threshold": -1,
            "endgame_max_hand_threshold": -1,
        }
    model.eval()
    losses = []
    maes = []
    end_abs_errs = []
    end_sign_ok = []
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            chunk = data[i : i + batch_size]
            x, y = sample_batch(chunk, len(chunk))
            x = x.to(device)
            y = y.to(device)
            pred = model(x).view(-1)
            losses.append(nn.functional.mse_loss(pred, y).item())
            abs_err = torch.abs(pred - y)
            maes.append(torch.mean(abs_err).item())
            for j, sample in enumerate(chunk):
                if _is_endgame_sample(sample, end_cfg):
                    pa = float(pred[j].item())
                    ya = float(y[j].item())
                    end_abs_errs.append(abs(pa - ya))
                    end_sign_ok.append((pa >= 0.0) == (ya >= 0.0))
    end_mae = float(np.mean(end_abs_errs)) if end_abs_errs else None
    end_sign_acc = float(np.mean(end_sign_ok)) if end_sign_ok else None
    return float(np.mean(losses)), float(np.mean(maes)), end_mae, end_sign_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--updates-per-episode", type=int, default=2)
    parser.add_argument("--val-size", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--value-scale", type=float, default=15.0)
    parser.add_argument("--save", default="ML_SKHU/models/value/value_heuristic_fullinfo.pt")
    parser.add_argument("--init-ckpt", default="")
    parser.add_argument("--buffer-capacity", type=int, default=100000)
    parser.add_argument("--endgame-steps-threshold", type=int, default=8)
    parser.add_argument("--endgame-total-threshold", type=int, default=-1)
    parser.add_argument("--endgame-max-hand-threshold", type=int, default=-1)
    parser.add_argument("--endgame-oversample", type=int, default=1)  # legacy fallback
    parser.add_argument("--steps-weight-le8-steps", type=int, default=8)
    parser.add_argument("--steps-weight-le16-steps", type=int, default=16)
    parser.add_argument("--steps-weight-le24-steps", type=int, default=24)
    parser.add_argument("--steps-weight-le8", type=int, default=4)
    parser.add_argument("--steps-weight-le16", type=int, default=3)
    parser.add_argument("--steps-weight-le24", type=int, default=2)
    parser.add_argument("--steps-weight-gt24", type=int, default=1)
    parser.add_argument("--save-best", action="store_true")
    parser.add_argument("--save-best-val-mae", default="")
    parser.add_argument("--save-best-endgame-mae", default="")
    args = parser.parse_args()

    # infer input dim from actual encoder
    g = big2Game.big2Game()
    p_dim = int(encode_policy_input(g, 1, _oracle_belief_probs(g)).shape[0])

    model = ValueModel(p_dim, hidden_dim=512).to(args.device)
    if args.init_ckpt:
        model.load_state_dict(torch.load(args.init_ckpt, map_location=args.device))

    opt = optim.Adam(model.parameters(), lr=args.lr)
    train_data = []
    val_data = None
    best_val_mae = None
    best_end_mae = None

    end_cfg = {
        "endgame_steps_threshold": args.endgame_steps_threshold,
        "endgame_total_threshold": args.endgame_total_threshold,
        "endgame_max_hand_threshold": args.endgame_max_hand_threshold,
    }
    step_weight_cfg = {
        "w_le8_steps": args.steps_weight_le8_steps,
        "w_le16_steps": args.steps_weight_le16_steps,
        "w_le24_steps": args.steps_weight_le24_steps,
        "w_le8": args.steps_weight_le8,
        "w_le16": args.steps_weight_le16,
        "w_le24": args.steps_weight_le24,
        "w_gt24": args.steps_weight_gt24,
    }
    save_best_val_path = args.save_best_val_mae or ""
    save_best_end_path = args.save_best_endgame_mae or ""
    if args.save_best:
        base_root, base_ext = os.path.splitext(args.save)
        if not save_best_val_path:
            save_best_val_path = f"{base_root}.best_val_mae{base_ext}"
        if not save_best_end_path:
            save_best_end_path = f"{base_root}.best_endgame_mae{base_ext}"

    for ep in range(1, args.episodes + 1):
        ep_samples = collect_episode_samples(value_scale=args.value_scale)
        for s in ep_samples:
            factor = _step_bucket_oversample_factor(s, step_weight_cfg)
            # Legacy fallback path if all ladder weights are effectively x1.
            if (
                step_weight_cfg["w_le8"] == 1
                and step_weight_cfg["w_le16"] == 1
                and step_weight_cfg["w_le24"] == 1
                and step_weight_cfg["w_gt24"] == 1
                and args.endgame_oversample > 1
                and _is_endgame_sample(s, end_cfg)
            ):
                factor = int(args.endgame_oversample)
            for _ in range(max(1, factor)):
                train_data.append(s)
        if len(train_data) > args.buffer_capacity:
            train_data = train_data[-args.buffer_capacity :]

        if val_data is None and len(train_data) >= args.val_size:
            idx = np.random.choice(len(train_data), size=args.val_size, replace=False)
            val_data = [train_data[int(i)] for i in idx]

        loss_val = None
        for _ in range(args.updates_per_episode):
            if not train_data:
                continue
            x, y = sample_batch(train_data, args.batch_size)
            x = x.to(args.device)
            y = y.to(args.device)
            model.train()
            pred = model(x).view(-1)
            loss = nn.functional.mse_loss(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_val = float(loss.item())

        if ep % args.log_every == 0 or ep == 1:
            if val_data is not None:
                val_loss, val_mae, end_mae, end_sign_acc = eval_value(
                    model,
                    val_data,
                    device=args.device,
                    end_cfg=end_cfg,
                )
            else:
                val_loss, val_mae, end_mae, end_sign_acc = (None, None, None, None)
            loss_str = f"{loss_val:.4f}" if loss_val is not None else "n/a"
            vl_str = f"{val_loss:.4f}" if val_loss is not None else "n/a"
            mae_str = f"{val_mae:.4f}" if val_mae is not None else "n/a"
            end_mae_str = f"{end_mae:.4f}" if end_mae is not None else "n/a"
            end_sign_str = f"{end_sign_acc:.4f}" if end_sign_acc is not None else "n/a"
            print(
                f"episode={ep}/{args.episodes} "
                f"value_loss={loss_str} val_loss={vl_str} val_mae={mae_str} "
                f"endgame_val_mae={end_mae_str} endgame_sign_acc={end_sign_str} "
                f"buffer={len(train_data)} ep_samples={len(ep_samples)}"
            )

            if args.save_best and val_mae is not None:
                if best_val_mae is None or val_mae < best_val_mae:
                    best_val_mae = float(val_mae)
                    os.makedirs(os.path.dirname(save_best_val_path), exist_ok=True)
                    torch.save(model.state_dict(), save_best_val_path)
                    print(
                        f"saved best_val_mae checkpoint: {save_best_val_path} "
                        f"(ep={ep} val_mae={best_val_mae:.4f})"
                    )
                if end_mae is not None and (best_end_mae is None or end_mae < best_end_mae):
                    best_end_mae = float(end_mae)
                    os.makedirs(os.path.dirname(save_best_end_path), exist_ok=True)
                    torch.save(model.state_dict(), save_best_end_path)
                    print(
                        f"saved best_endgame_mae checkpoint: {save_best_end_path} "
                        f"(ep={ep} endgame_val_mae={best_end_mae:.4f})"
                    )

    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    torch.save(model.state_dict(), args.save)
    print(f"saved value model: {args.save}")


if __name__ == "__main__":
    main()
