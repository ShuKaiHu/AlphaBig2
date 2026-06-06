import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import enumerateOptions

from ML_AB.actions import ACTION_DIM, action_features_torch, action_to_string
from ML_AB.data import Replay
from ML_AB.eval import load_model
from ML_AB.models import checkpoint_payload
from ML_AB.online import build_public_game
from ML_AB.state import action_mask, encode_game, public_belief_prior
from ML_AB.train import batch_to_tensors, train_step
from ML_AB.utils import configure_torch_threads, device_from_arg, set_seed


BIG2_TO_ALPHA_SUIT = {"1": 4, "2": 3, "3": 2, "4": 1}
BIG2_TO_ALPHA_RANK = {
    "3": 1,
    "4": 2,
    "5": 3,
    "6": 4,
    "7": 5,
    "8": 6,
    "9": 7,
    "T": 8,
    "J": 9,
    "Q": 10,
    "K": 11,
    "1": 12,
    "2": 13,
}


def _default_dataset_path():
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root.parent / "Big2VisionAgent-codex" / "data" / "live_training_corpus.jsonl"


def _card_code_to_alpha_id(code):
    text = str(code)
    return (BIG2_TO_ALPHA_RANK[text[1]] - 1) * 4 + BIG2_TO_ALPHA_SUIT[text[0]]


def _action_index_from_ref(ref):
    if isinstance(ref.get("action_index"), int):
        return int(ref["action_index"])
    if ref.get("action") == "pass":
        return int(enumerateOptions.passInd)
    codes = list(ref.get("card_codes") or [])
    if not codes:
        raise ValueError(f"missing card_codes in action ref: {ref}")
    cards = sorted(_card_code_to_alpha_id(code) for code in codes)
    return int(enumerateOptions.action_index_from_cards(cards))


def _load_rows(paths):
    rows = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row.get("ml_public_state"), dict):
                    continue
                label = row.get("preference_label")
                if not isinstance(label, dict):
                    continue
                if not label.get("preferred_actions") or not label.get("bad_actions"):
                    continue
                rows.append(row)
    return rows


def _sample_from_preference_row(row, args):
    game = build_public_game(**row["ml_public_state"])
    player = int(game.playersGo)
    encoded = encode_game(game, player, public_belief_prior(game, player))
    mask = action_mask(game)
    label = row["preference_label"]

    good = []
    for ref in label.get("preferred_actions") or []:
        action = _action_index_from_ref(ref)
        if 0 <= action < ACTION_DIM and mask[action] > 0:
            good.append(int(action))
    bad = []
    for ref in label.get("bad_actions") or []:
        action = _action_index_from_ref(ref)
        if 0 <= action < ACTION_DIM and mask[action] > 0:
            bad.append(int(action))
    good = sorted(set(good))
    bad = sorted(set(bad))
    if not good or not bad:
        raise ValueError(f"no valid preference actions for {row.get('decision_id')}")

    policy_target = np.zeros((ACTION_DIM,), dtype=np.float32)
    for action in good:
        policy_target[action] = 1.0 / float(len(good))

    score = row.get("round_self_score")
    value_target = 0.0
    if isinstance(score, int):
        value_target = float(np.tanh(float(score) / float(args.value_scale)))

    return {
        "card_feats": encoded["card_feats"],
        "history_feats": encoded["history_feats"],
        "global_feats": encoded["global_feats"],
        "action_mask": mask,
        "action": int(good[0]),
        "policy_target": policy_target,
        "player": player,
        "value_target": value_target,
        "value_sample_weight": 1.0,
        "belief_target": np.full((52,), -1, dtype=np.int64),
        "belief_mask": np.zeros((52,), dtype=np.float32),
        "repair_good_actions": good,
        "repair_bad_actions": bad,
        "source_decision_id": row.get("decision_id"),
        "preference_reason": label.get("reason"),
    }


def build_preference_samples(rows, args):
    samples = []
    skipped = []
    for row in rows:
        try:
            samples.append(_sample_from_preference_row(row, args))
        except Exception as exc:
            skipped.append({"decision_id": row.get("decision_id"), "error": str(exc)})
    return samples, skipped


def _set_policy_head_only(model):
    prefixes = ("action_proj.", "action_bias.", "logit_scale")
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith(prefixes)


def _preference_metrics(model, samples, device, margin, limit=0):
    selected = samples
    if limit and int(limit) > 0:
        selected = selected[: int(limit)]
    if not selected:
        return {}
    model.eval()
    wins = 0
    violations = []
    with torch.no_grad():
        for sample in selected:
            card = torch.tensor(sample["card_feats"], dtype=torch.float32, device=device).unsqueeze(0)
            hist = torch.tensor(sample["history_feats"], dtype=torch.float32, device=device).unsqueeze(0)
            glob = torch.tensor(sample["global_feats"], dtype=torch.float32, device=device).unsqueeze(0)
            af = action_features_torch(device).unsqueeze(0)
            mask = torch.tensor(sample["action_mask"], dtype=torch.float32, device=device).unsqueeze(0)
            logits, _value, _belief = model(card, hist, glob, af, mask)
            good = torch.tensor(sample["repair_good_actions"], dtype=torch.long, device=device)
            bad = torch.tensor(sample["repair_bad_actions"], dtype=torch.long, device=device)
            good_max = logits[0, good].max()
            bad_max = logits[0, bad].max()
            wins += int(good_max > bad_max)
            violations.append(float(F.relu(bad_max - good_max + float(margin)).cpu().item()))
    return {
        "preference_accuracy": float(wins / float(len(selected))),
        "preference_margin_loss": float(np.mean(violations)),
    }


def preference_train_step(model, optimizer, batch, device, args):
    row = train_step(
        model,
        optimizer,
        batch,
        device,
        policy_weight=args.policy_weight,
        value_weight=args.value_weight,
        belief_weight=0.0,
    )
    if args.margin_weight <= 0:
        row["repair_margin_loss"] = 0.0
        return row

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
        _value_sample_weight,
        _btarget,
        _bmask,
        af,
    ) = batch_to_tensors(batch, device)
    logits, _pred_value, _belief_logits = model(card, hist, glob, af, mask)
    losses = []
    for index, sample in enumerate(batch):
        good = [int(action) for action in sample.get("repair_good_actions", []) if mask[index, int(action)] > 0.5]
        bad = [int(action) for action in sample.get("repair_bad_actions", []) if mask[index, int(action)] > 0.5]
        if not good or not bad:
            continue
        good_t = torch.tensor(good, dtype=torch.long, device=device)
        bad_t = torch.tensor(bad, dtype=torch.long, device=device)
        losses.append(F.relu(logits[index, bad_t].max() - logits[index, good_t].max() + float(args.margin)))
    if not losses:
        row["repair_margin_loss"] = 0.0
        return row

    margin_loss = torch.stack(losses).mean()
    optimizer.zero_grad()
    (float(args.margin_weight) * margin_loss).backward()
    torch.nn.utils.clip_grad_norm_([param for param in model.parameters() if param.requires_grad], 1.0)
    optimizer.step()
    row["repair_margin_loss"] = float(margin_loss.item())
    return row


def _split_samples(samples, eval_fraction, seed):
    if len(samples) <= 1 or eval_fraction <= 0:
        return list(samples), []
    rng = np.random.default_rng(int(seed) + 3571)
    indices = rng.permutation(len(samples))
    eval_count = int(round(len(samples) * float(eval_fraction)))
    eval_count = max(1, min(eval_count, len(samples) - 1))
    eval_idx = set(int(idx) for idx in indices[:eval_count])
    train = [sample for idx, sample in enumerate(samples) if idx not in eval_idx]
    eval_samples = [sample for idx, sample in enumerate(samples) if idx in eval_idx]
    return train, eval_samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default=str(_default_dataset_path()))
    parser.add_argument("--init", default="ML_AB/models/candidate_value_head_live_mcts.pt")
    parser.add_argument("--save", default="ML_AB/models/candidate_preference_repair.pt")
    parser.add_argument("--metrics", default="ML_AB/runs/live_preference_repair.jsonl")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--replay-repeat", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--policy-weight", type=float, default=1.0)
    parser.add_argument("--value-weight", type=float, default=0.0)
    parser.add_argument("--margin", type=float, default=0.35)
    parser.add_argument("--margin-weight", type=float, default=1.0)
    parser.add_argument("--policy-head-only", action="store_true")
    parser.add_argument("--eval-fraction", type=float, default=0.0)
    parser.add_argument("--eval-limit", type=int, default=512)
    parser.add_argument("--value-scale", type=float, default=15.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=260603)
    args = parser.parse_args()

    configure_torch_threads()
    set_seed(args.seed)
    device = device_from_arg(args.device)
    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    os.makedirs(os.path.dirname(args.metrics), exist_ok=True)

    dataset_paths = [Path(item).expanduser() for item in str(args.datasets).split(",") if item.strip()]
    rows = _load_rows(dataset_paths)
    samples, skipped = build_preference_samples(rows, args)
    if not samples:
        raise ValueError(f"no preference samples loaded from {dataset_paths}; skipped={skipped[:5]}")
    train_samples, eval_samples = _split_samples(samples, args.eval_fraction, args.seed)

    model = load_model(args.init, device)
    if args.policy_head_only:
        _set_policy_head_only(model)
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-5)

    replay = Replay(capacity=max(len(train_samples) * int(args.replay_repeat), int(args.batch_size)))
    for _ in range(max(int(args.replay_repeat), 1)):
        replay.add_many(train_samples)

    before_train = _preference_metrics(model, train_samples, device, args.margin, args.eval_limit)
    before_eval = _preference_metrics(model, eval_samples, device, args.margin, args.eval_limit)
    reason_counts = {}
    for sample in samples:
        reason = sample.get("preference_reason")
        reason_counts[reason] = reason_counts.get(reason, 0) + 1

    metrics_file = open(args.metrics, "a", buffering=1)
    t0 = time.time()
    last = {}
    for step in range(1, int(args.steps) + 1):
        model.train()
        last = preference_train_step(model, optimizer, replay.sample(args.batch_size), device, args)
        row = {
            "step": step,
            "samples": len(samples),
            "train_samples": len(train_samples),
            "eval_samples": len(eval_samples),
            "skipped": len(skipped),
            "reason_counts": reason_counts,
            "policy_head_only": bool(args.policy_head_only),
            "trainable_params": int(sum(param.numel() for param in trainable_params)),
            "elapsed_sec": round(time.time() - t0, 3),
            **last,
        }
        metrics_file.write(json.dumps(row, sort_keys=True) + "\n")
        if step == 1 or step % 25 == 0:
            print(json.dumps(row, sort_keys=True))

    after_train = _preference_metrics(model, train_samples, device, args.margin, args.eval_limit)
    after_eval = _preference_metrics(model, eval_samples, device, args.margin, args.eval_limit)
    payload = checkpoint_payload(
        model,
        vars(args),
        {
            "last": last,
            "samples": len(samples),
            "train_samples": len(train_samples),
            "eval_samples": len(eval_samples),
            "skipped": skipped[:20],
            "reason_counts": reason_counts,
            "policy_head_only": bool(args.policy_head_only),
            "trainable_params": int(sum(param.numel() for param in trainable_params)),
            "preference_before_train": before_train,
            "preference_after_train": after_train,
            "preference_before_eval": before_eval,
            "preference_after_eval": after_eval,
        },
    )
    torch.save(payload, args.save)
    print(
        json.dumps(
            {
                "saved": args.save,
                "samples": len(samples),
                "train_samples": len(train_samples),
                "eval_samples": len(eval_samples),
                "skipped": len(skipped),
                "reason_counts": reason_counts,
                "preference_before_train": before_train,
                "preference_after_train": after_train,
                "preference_before_eval": before_eval,
                "preference_after_eval": after_eval,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
