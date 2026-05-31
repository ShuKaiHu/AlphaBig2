import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from ML_AB.actions import ACTION_DIM, action_cards, action_to_string
from ML_AB.data import Replay
from ML_AB.eval import load_model
from ML_AB.online import build_public_game
from ML_AB.models import checkpoint_payload
from ML_AB.state import action_mask, belief_targets, encode_game, public_belief_prior
from ML_AB.train import batch_to_tensors, train_step
from ML_AB.utils import configure_torch_threads, device_from_arg, set_seed


def _sample_from_observation(observation, target_actions, value_target=0.0):
    game = build_public_game(**observation)
    player = int(game.playersGo)
    b_prior = public_belief_prior(game, player)
    encoded = encode_game(game, player, b_prior)
    mask = action_mask(game)
    b_target, b_mask = belief_targets(game, player)

    target = np.zeros((ACTION_DIM,), dtype=np.float32)
    valid_targets = []
    for action in target_actions:
        if 0 <= int(action) < ACTION_DIM and mask[int(action)] > 0:
            valid_targets.append(int(action))
    if not valid_targets:
        raise ValueError(f"no valid target actions for hand={observation.get('my_hand')}")
    for action in valid_targets:
        target[action] = 1.0 / float(len(valid_targets))

    return {
        "card_feats": encoded["card_feats"],
        "history_feats": encoded["history_feats"],
        "global_feats": encoded["global_feats"],
        "action_mask": mask,
        "action": int(valid_targets[0]),
        "policy_target": target,
        "value_target": float(value_target),
        "player": player,
        "belief_target": b_target,
        "belief_mask": b_mask,
        "repair_good_actions": valid_targets,
    }


def _find_actions_by_cards(cards):
    wanted = sorted(int(c) for c in cards)
    return [a for a in range(ACTION_DIM) if sorted(action_cards(a)) == wanted]


def _target_actions_from_specs(specs):
    actions = []
    for cards in specs:
        matches = _find_actions_by_cards(cards)
        if not matches:
            raise ValueError(f"target cards not found in action space: {cards}")
        actions.extend(matches)
    return actions


def _load_live_observations(path):
    rows = {}
    with open(path, "r") as f:
        for idx, line in enumerate(f, 1):
            row = json.loads(line)
            rows[idx] = row
    return rows


def build_repair_samples(model_debug_jsonl):
    rows = _load_live_observations(model_debug_jsonl)
    repairs = []

    # Row 14: with only 4H and QD left under control, unload the lower single first.
    repairs.append(
        (
            "endgame_4_before_q",
            rows[14]["ml_public_state"],
            _target_actions_from_specs([[7]]),
            _target_actions_from_specs([[38]]),
        )
    )

    # Row 24: avoid spending 6H because it destroys the 4-5-6-7-8 straight.
    # Either 8C or 8D keeps a straight available through the duplicate eight.
    repairs.append(
        (
            "preserve_straight_use_duplicate_8",
            rows[24]["ml_public_state"],
            _target_actions_from_specs([[21], [22]]),
            _target_actions_from_specs([[15], [20], [9]]),
        )
    )

    # Row 27: under control, avoid opening with the pair of twos when cheap singles
    # and a low pair are available.
    repairs.append(
        (
            "avoid_opening_pair_twos",
            rows[27]["ml_public_state"],
            _target_actions_from_specs([[2], [5], [6], [5, 6]]),
            _target_actions_from_specs([[49, 52]]),
        )
    )

    samples = []
    summary = []
    for name, obs, actions, avoid_actions in repairs:
        sample = _sample_from_observation(obs, actions)
        sample["repair_bad_actions"] = [
            int(a) for a in avoid_actions if 0 <= int(a) < ACTION_DIM and sample["action_mask"][int(a)] > 0
        ]
        samples.append(sample)
        summary.append(
            {
                "name": name,
                "targets": [action_to_string(a) for a in actions if sample["action_mask"][a] > 0],
                "avoid": [action_to_string(a) for a in sample["repair_bad_actions"]],
            }
        )
    return samples, summary


def repair_train_step(model, optimizer, batch, device, args):
    base = train_step(
        model,
        optimizer,
        batch,
        device,
        value_weight=args.value_weight,
        belief_weight=args.belief_weight,
    )
    if args.margin_weight <= 0:
        return base

    (
        card,
        hist,
        glob,
        mask,
        _action,
        _policy_target,
        _policy_target_mask,
        _policy_advantage,
        _policy_gradient_mask,
        _value,
        _btarget,
        _bmask,
        af,
    ) = batch_to_tensors(batch, device)
    logits, _pred_value, _belief_logits = model(card, hist, glob, af, mask)
    margin_losses = []
    for i, sample in enumerate(batch):
        good = [int(a) for a in sample.get("repair_good_actions", []) if mask[i, int(a)] > 0.5]
        bad = [int(a) for a in sample.get("repair_bad_actions", []) if mask[i, int(a)] > 0.5]
        if not good or not bad:
            continue
        good_scores = logits[i, torch.tensor(good, dtype=torch.long, device=device)]
        bad_scores = logits[i, torch.tensor(bad, dtype=torch.long, device=device)]
        margin_losses.append(F.relu(bad_scores.max() - good_scores.max() + float(args.margin)))
    if not margin_losses:
        base["repair_margin_loss"] = 0.0
        return base

    margin_loss = torch.stack(margin_losses).mean()
    loss = float(args.margin_weight) * margin_loss
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    base["repair_margin_loss"] = float(margin_loss.item())
    return base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", default="ML_AB/models/big2_transformer_best.pt")
    parser.add_argument("--save", default="ML_AB/models/candidate_live_repair.pt")
    parser.add_argument("--metrics", default="ML_AB/runs/live_repair.jsonl")
    parser.add_argument(
        "--model-debug-jsonl",
        default="/Users/shukaihu/Code_Project_Local/Big2VisionAgent-codex/artifacts/20260527-232233/autoplay_agent/model_debug.jsonl",
    )
    parser.add_argument("--steps", type=int, default=220)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--repeat", type=int, default=16)
    parser.add_argument("--lr", type=float, default=8e-7)
    parser.add_argument("--margin", type=float, default=0.35)
    parser.add_argument("--margin-weight", type=float, default=0.0)
    parser.add_argument("--only-repair", default="")
    parser.add_argument("--value-weight", type=float, default=0.0)
    parser.add_argument("--belief-weight", type=float, default=0.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=260539)
    args = parser.parse_args()

    configure_torch_threads()
    set_seed(args.seed)
    device = device_from_arg(args.device)
    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    os.makedirs(os.path.dirname(args.metrics), exist_ok=True)

    model = load_model(args.init, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    repair_samples, summary = build_repair_samples(args.model_debug_jsonl)
    if args.only_repair:
        keep = [i for i, row in enumerate(summary) if args.only_repair in row["name"]]
        repair_samples = [repair_samples[i] for i in keep]
        summary = [summary[i] for i in keep]
        if not repair_samples:
            raise ValueError(f"no repair matched --only-repair={args.only_repair!r}")
    replay = Replay(args.repeat * len(repair_samples))
    for _ in range(args.repeat):
        replay.add_many(repair_samples)

    metrics_file = open(args.metrics, "a", buffering=1)
    t0 = time.time()
    row = {}
    for step in range(1, args.steps + 1):
        model.train()
        row = repair_train_step(
            model,
            optimizer,
            replay.sample(args.batch_size),
            device,
            args,
        )
        row = {
            "step": step,
            "elapsed_sec": round(time.time() - t0, 3),
            "repair_samples": len(repair_samples),
            **row,
        }
        metrics_file.write(json.dumps(row, sort_keys=True) + "\n")
        if step == 1 or step % 25 == 0:
            print(json.dumps(row, sort_keys=True))

    payload = checkpoint_payload(model, vars(args), {"last": row, "repairs": summary})
    torch.save(payload, args.save)
    print(json.dumps({"repairs": summary}, ensure_ascii=False, sort_keys=True))
    print(f"saved checkpoint: {args.save}")


if __name__ == "__main__":
    main()
