import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

import enumerateOptions

from ML_AB.data import Replay
from ML_AB.eval import load_model
from ML_AB.live_belief_data import default_live_corpus_path
from ML_AB.models import checkpoint_payload
from ML_AB.online import build_public_game
from ML_AB.state import action_mask, encode_game, public_belief_prior
from ML_AB.train import train_step
from ML_AB.utils import configure_torch_threads, device_from_arg, set_seed


def _default_corpus_path():
    return default_live_corpus_path()


def _load_rows(path):
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("training_usable") is False:
                continue
            if not isinstance(row.get("round_self_score"), int):
                continue
            if not isinstance(row.get("ml_public_state"), dict):
                continue
            rows.append(row)
    return rows


def _row_agent_type(row):
    model_debug = row.get("model_debug")
    if isinstance(model_debug, dict) and model_debug.get("agent_type"):
        return str(model_debug.get("agent_type"))
    agent_type = row.get("agent_type")
    if agent_type is None:
        return None
    return str(agent_type)


def _filter_rows(rows, args):
    if not args.agent_type:
        return rows
    wanted = {
        item.strip().lower()
        for item in str(args.agent_type).split(",")
        if item.strip()
    }
    if not wanted:
        return rows
    return [
        row
        for row in rows
        if (_row_agent_type(row) or "").strip().lower() in wanted
    ]


def _split_samples(samples, eval_fraction, seed):
    if len(samples) <= 1 or eval_fraction <= 0:
        return list(samples), []
    rng = np.random.default_rng(int(seed) + 7919)
    indices = rng.permutation(len(samples))
    eval_count = int(round(len(samples) * float(eval_fraction)))
    eval_count = max(1, min(eval_count, len(samples) - 1))
    eval_idx = set(int(idx) for idx in indices[:eval_count])
    train = [sample for idx, sample in enumerate(samples) if idx not in eval_idx]
    eval_samples = [sample for idx, sample in enumerate(samples) if idx in eval_idx]
    return train, eval_samples


def _decision_action_index(row):
    decision = row.get("decision") or {}
    if decision.get("action") == "pass":
        return int(enumerateOptions.passInd)
    wanted_cards = list(decision.get("card_codes") or [])
    wanted_combo = decision.get("combo_type")
    for candidate in row.get("candidate_scores") or []:
        if candidate.get("action") != decision.get("action"):
            continue
        if candidate.get("combo_type") != wanted_combo:
            continue
        if list(candidate.get("card_codes") or []) != wanted_cards:
            continue
        action_index = candidate.get("action_index")
        if isinstance(action_index, int):
            return int(action_index)
    raise ValueError(f"could not map decision to action index: {decision}")


def _score_weight(score, big_loss_threshold, big_loss_weight):
    if score >= 0:
        return 1.0
    if abs(score) < big_loss_threshold:
        return 1.0
    scaled = min(abs(float(score)) / max(float(big_loss_threshold), 1.0), 3.0)
    return 1.0 + float(big_loss_weight) * scaled


def _belief_arrays_from_row(row):
    target = row.get("belief_target")
    mask = row.get("belief_mask")
    if not isinstance(target, list) or not isinstance(mask, list):
        return (
            np.full((52,), -1, dtype=np.int64),
            np.zeros((52,), dtype=np.float32),
        )
    if len(target) != 52 or len(mask) != 52:
        return (
            np.full((52,), -1, dtype=np.int64),
            np.zeros((52,), dtype=np.float32),
        )
    target_arr = np.asarray(target, dtype=np.int64)
    mask_arr = np.asarray(mask, dtype=np.float32)
    valid = (mask_arr > 0.5) & (target_arr >= 0) & (target_arr < 3)
    clean_target = np.full((52,), -1, dtype=np.int64)
    clean_mask = np.zeros((52,), dtype=np.float32)
    clean_target[valid] = target_arr[valid]
    clean_mask[valid] = 1.0
    return clean_target, clean_mask


def _sample_from_live_row(row, args):
    game = build_public_game(**row["ml_public_state"])
    player = int(game.playersGo)
    encoded = encode_game(game, player, public_belief_prior(game, player))
    mask = action_mask(game)
    action = _decision_action_index(row)
    if not (0 <= action < mask.shape[0]):
        raise ValueError(f"action {action} is outside action space for {row.get('decision_id')}")
    mask_repaired = False
    if mask[action] <= 0:
        mask = mask.copy()
        mask[action] = 1.0
        mask_repaired = True

    belief_target, belief_mask = _belief_arrays_from_row(row)
    if args.require_belief_labels and float(belief_mask.sum()) <= 0:
        raise ValueError(f"missing belief labels for {row.get('decision_id')}")

    score = int(row["round_self_score"])
    sample = {
        "card_feats": encoded["card_feats"],
        "history_feats": encoded["history_feats"],
        "global_feats": encoded["global_feats"],
        "action_mask": mask,
        "action": int(action),
        "player": player,
        "value_target": float(np.tanh(float(score) / float(args.value_scale))),
        "value_sample_weight": _score_weight(score, args.big_loss_threshold, args.big_loss_weight),
        "belief_target": belief_target,
        "belief_mask": belief_mask,
        "policy_advantage": 0.0,
        "mask_repaired": mask_repaired,
        "source_decision_id": row.get("decision_id"),
        "source_score": score,
    }
    if args.policy_gradient and abs(score) >= args.policy_gradient_min_abs_score:
        sample["policy_gradient"] = True
    return sample


def build_live_samples(rows, args):
    samples = []
    skipped = []
    for row in rows:
        try:
            samples.append(_sample_from_live_row(row, args))
        except Exception as exc:
            skipped.append({"decision_id": row.get("decision_id"), "error": str(exc)})
    return samples, skipped


def _value_mae(model, samples, device, limit):
    if not samples:
        return None
    selected = samples[: min(len(samples), int(limit))]
    errors = []
    model.eval()
    with torch.no_grad():
        for sample in selected:
            card = torch.tensor(sample["card_feats"], dtype=torch.float32, device=device).unsqueeze(0)
            hist = torch.tensor(sample["history_feats"], dtype=torch.float32, device=device).unsqueeze(0)
            glob = torch.tensor(sample["global_feats"], dtype=torch.float32, device=device).unsqueeze(0)
            from ML_AB.actions import action_features_torch

            af = action_features_torch(device).unsqueeze(0)
            mask = torch.tensor(sample["action_mask"], dtype=torch.float32, device=device).unsqueeze(0)
            _logits, value, _belief = model(card, hist, glob, af, mask)
            errors.append(abs(float(value.cpu().numpy()[0]) - float(sample["value_target"])))
    return float(np.mean(errors))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default=str(_default_corpus_path()))
    parser.add_argument("--init", default="ML_AB/models/big2_transformer_best.pt")
    parser.add_argument("--save", default="ML_AB/models/candidate_live_corpus_value.pt")
    parser.add_argument("--metrics", default="ML_AB/runs/live_corpus_value.jsonl")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--replay-repeat", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=260601)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--value-scale", type=float, default=15.0)
    parser.add_argument("--policy-weight", type=float, default=1.0)
    parser.add_argument("--value-weight", type=float, default=1.0)
    parser.add_argument("--belief-weight", type=float, default=0.0)
    parser.add_argument("--require-belief-labels", action="store_true")
    parser.add_argument("--belief-head-only", action="store_true")
    parser.add_argument("--value-head-only", action="store_true")
    parser.add_argument("--agent-type", default="")
    parser.add_argument("--eval-fraction", type=float, default=0.15)
    parser.add_argument("--big-loss-threshold", type=float, default=8.0)
    parser.add_argument("--big-loss-weight", type=float, default=1.0)
    parser.add_argument("--policy-gradient", action="store_true")
    parser.add_argument("--policy-gradient-min-abs-score", type=float, default=3.0)
    parser.add_argument("--eval-limit", type=int, default=512)
    args = parser.parse_args()

    configure_torch_threads()
    set_seed(args.seed)
    device = device_from_arg(args.device)
    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    os.makedirs(os.path.dirname(args.metrics), exist_ok=True)
    if args.belief_head_only and args.value_head_only:
        raise ValueError("--belief-head-only and --value-head-only are mutually exclusive")

    loaded_rows = _load_rows(args.corpus)
    rows = _filter_rows(loaded_rows, args)
    samples, skipped = build_live_samples(rows, args)
    if not samples:
        raise ValueError(f"no usable live samples loaded from {args.corpus}")
    train_samples, eval_samples = _split_samples(samples, args.eval_fraction, args.seed)

    model = load_model(args.init, device)
    if args.belief_head_only:
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith("belief_head.")
    if args.value_head_only:
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith("value_head.")
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.lr,
        weight_decay=1e-5,
    )
    replay = Replay(capacity=max(len(train_samples) * args.replay_repeat, args.batch_size))
    for _ in range(max(int(args.replay_repeat), 1)):
        replay.add_many(train_samples)

    score_values = [sample["source_score"] for sample in train_samples]
    mask_repaired_count = sum(1 for sample in train_samples if sample.get("mask_repaired"))
    belief_labeled_count = sum(1 for sample in train_samples if float(np.sum(sample["belief_mask"])) > 0.0)
    metrics_file = open(args.metrics, "a", buffering=1)
    t0 = time.time()
    train_mae_before = _value_mae(model, train_samples, device, args.eval_limit)
    eval_mae_before = _value_mae(model, eval_samples, device, args.eval_limit) if eval_samples else None
    last = {}
    for step in range(1, args.steps + 1):
        model.train()
        last = train_step(
            model,
            optimizer,
            replay.sample(args.batch_size),
            device,
            policy_weight=args.policy_weight,
            value_weight=args.value_weight,
            belief_weight=args.belief_weight,
        )
        row = {
            "step": step,
            "samples": len(samples),
            "loaded_rows": len(loaded_rows),
            "filtered_rows": len(rows),
            "train_samples": len(train_samples),
            "eval_samples": len(eval_samples),
            "skipped": len(skipped),
            "score_mean": float(np.mean(score_values)),
            "score_min": int(np.min(score_values)),
            "score_max": int(np.max(score_values)),
            "mask_repaired": int(mask_repaired_count),
            "belief_labeled": int(belief_labeled_count),
            "belief_head_only": bool(args.belief_head_only),
            "value_head_only": bool(args.value_head_only),
            "agent_type_filter": str(args.agent_type),
            "trainable_params": int(trainable_params),
            "policy_gradient": bool(args.policy_gradient),
            "elapsed_sec": round(time.time() - t0, 3),
            **last,
        }
        metrics_file.write(json.dumps(row, sort_keys=True) + "\n")
        if step == 1 or step % 50 == 0:
            print(json.dumps(row, sort_keys=True))

    train_mae_after = _value_mae(model, train_samples, device, args.eval_limit)
    eval_mae_after = _value_mae(model, eval_samples, device, args.eval_limit) if eval_samples else None
    payload = checkpoint_payload(
        model,
        vars(args),
        {
            "last": last,
            "samples": len(samples),
            "loaded_rows": len(loaded_rows),
            "filtered_rows": len(rows),
            "train_samples": len(train_samples),
            "eval_samples": len(eval_samples),
            "skipped": skipped[:20],
            "mask_repaired": int(mask_repaired_count),
            "belief_labeled": int(belief_labeled_count),
            "belief_head_only": bool(args.belief_head_only),
            "value_head_only": bool(args.value_head_only),
            "agent_type_filter": str(args.agent_type),
            "trainable_params": int(trainable_params),
            "value_mae_before": eval_mae_before if eval_mae_before is not None else train_mae_before,
            "value_mae_after": eval_mae_after if eval_mae_after is not None else train_mae_after,
            "train_value_mae_before": train_mae_before,
            "train_value_mae_after": train_mae_after,
            "eval_value_mae_before": eval_mae_before,
            "eval_value_mae_after": eval_mae_after,
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
                "mask_repaired": int(mask_repaired_count),
                "train_value_mae_before": train_mae_before,
                "train_value_mae_after": train_mae_after,
                "eval_value_mae_before": eval_mae_before,
                "eval_value_mae_after": eval_mae_after,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
