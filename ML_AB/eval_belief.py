import argparse
import json
from collections import defaultdict

import numpy as np
import torch

from ML_AB.actions import action_features_torch
from ML_AB.eval import load_model
from ML_AB.live_belief_data import default_live_belief_dataset_path, load_jsonl
from ML_AB.online import build_public_game
from ML_AB.state import action_mask, encode_game, public_belief_prior
from ML_AB.utils import configure_torch_threads, device_from_arg


EPS = 1.0e-9


def _softmax(values):
    values = np.asarray(values, dtype=np.float64)
    values = values - np.max(values, axis=-1, keepdims=True)
    out = np.exp(values)
    out /= np.maximum(out.sum(axis=-1, keepdims=True), EPS)
    return out


def _empty_bucket():
    return {
        "states": 0,
        "cards": 0,
        "model_nll_sum": 0.0,
        "prior_nll_sum": 0.0,
        "uniform_nll_sum": 0.0,
        "model_top1": 0,
        "prior_top1": 0,
        "model_true_prob_sum": 0.0,
        "prior_true_prob_sum": 0.0,
    }


def _add_metrics(bucket, model_probs, prior_probs, targets):
    if len(targets) == 0:
        return
    indices = np.arange(len(targets))
    model_true = np.maximum(model_probs[indices, targets], EPS)
    prior_true = np.maximum(prior_probs[indices, targets], EPS)
    bucket["states"] += 1
    bucket["cards"] += int(len(targets))
    bucket["model_nll_sum"] += float(-np.log(model_true).sum())
    bucket["prior_nll_sum"] += float(-np.log(prior_true).sum())
    bucket["uniform_nll_sum"] += float(-np.log(np.full_like(model_true, 1.0 / 3.0)).sum())
    bucket["model_top1"] += int((np.argmax(model_probs, axis=-1) == targets).sum())
    bucket["prior_top1"] += int((np.argmax(prior_probs, axis=-1) == targets).sum())
    bucket["model_true_prob_sum"] += float(model_true.sum())
    bucket["prior_true_prob_sum"] += float(prior_true.sum())


def _finalize_bucket(bucket):
    cards = int(bucket["cards"])
    if cards <= 0:
        return None
    return {
        "states": int(bucket["states"]),
        "cards": cards,
        "model_nll": float(bucket["model_nll_sum"] / cards),
        "prior_nll": float(bucket["prior_nll_sum"] / cards),
        "uniform_nll": float(bucket["uniform_nll_sum"] / cards),
        "model_top1": float(bucket["model_top1"] / cards),
        "prior_top1": float(bucket["prior_top1"] / cards),
        "model_true_prob": float(bucket["model_true_prob_sum"] / cards),
        "prior_true_prob": float(bucket["prior_true_prob_sum"] / cards),
    }


def _belief_prediction(model, row, device):
    game = build_public_game(**row["ml_public_state"])
    prior = public_belief_prior(game, 1)
    state = encode_game(game, 1, prior)
    with torch.no_grad():
        card = torch.tensor(state["card_feats"], dtype=torch.float32, device=device).unsqueeze(0)
        hist = torch.tensor(state["history_feats"], dtype=torch.float32, device=device).unsqueeze(0)
        glob = torch.tensor(state["global_feats"], dtype=torch.float32, device=device).unsqueeze(0)
        af = action_features_torch(device).unsqueeze(0)
        mask = torch.tensor(action_mask(game), dtype=torch.float32, device=device).unsqueeze(0)
        _logits, _value, belief_logits = model(card, hist, glob, af, mask)
    return _softmax(belief_logits.cpu().numpy()[0]), prior


def evaluate_belief(model, rows, device, limit=0):
    buckets = defaultdict(_empty_bucket)
    skipped = []
    used_rows = rows[: int(limit)] if int(limit) > 0 else rows
    for row in used_rows:
        try:
            target = np.asarray(row["belief_target"], dtype=np.int64)
            belief_mask = np.asarray(row["belief_mask"], dtype=np.float32) > 0.5
            valid = belief_mask & (target >= 0) & (target < 3)
            if not bool(valid.any()):
                raise ValueError("no belief labels")
            model_probs, prior = _belief_prediction(model, row, device)
        except Exception as exc:
            skipped.append({"decision_id": row.get("decision_id"), "reason": str(exc)})
            continue

        card_indices = np.flatnonzero(valid)
        labels = target[card_indices]
        model_selected = model_probs[card_indices]
        prior_selected = prior[card_indices]
        _add_metrics(buckets["all"], model_selected, prior_selected, labels)
        _add_metrics(
            buckets[f"played/{row.get('belief_phase_by_played', 'unknown')}"],
            model_selected,
            prior_selected,
            labels,
        )
        _add_metrics(
            buckets[f"unknown/{row.get('belief_phase_by_unknown', 'unknown')}"],
            model_selected,
            prior_selected,
            labels,
        )

    summary = {
        key: _finalize_bucket(value)
        for key, value in sorted(buckets.items())
        if _finalize_bucket(value) is not None
    }
    return summary, skipped


def gate_result(summary):
    all_metrics = summary.get("all") or {}
    mid = summary.get("unknown/mid_15_29_unknown") or {}
    late = summary.get("unknown/late_0_14_unknown") or {}
    checks = {
        "overall_nll_beats_prior": all_metrics.get("model_nll", float("inf"))
        < all_metrics.get("prior_nll", -float("inf")),
        "mid_nll_beats_prior": mid.get("model_nll", float("inf"))
        < mid.get("prior_nll", -float("inf")),
        "late_nll_beats_prior": late.get("model_nll", float("inf"))
        < late.get("prior_nll", -float("inf")),
        "late_top1_beats_prior": late.get("model_top1", -float("inf"))
        > late.get("prior_top1", float("inf")),
    }
    return {"passed": all(checks.values()), "checks": checks}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="ML_AB/models/big2_transformer_best.pt")
    parser.add_argument("--dataset", default=str(default_live_belief_dataset_path()))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--save", default="")
    parser.add_argument("--fail-if-not-better", action="store_true")
    args = parser.parse_args()

    configure_torch_threads()
    device = device_from_arg(args.device)
    rows = load_jsonl(args.dataset)
    model = load_model(args.ckpt, device)
    summary, skipped = evaluate_belief(model, rows, device, args.limit)
    output = {
        "ckpt": args.ckpt,
        "dataset": args.dataset,
        "rows": len(rows),
        "skipped": len(skipped),
        "metrics": summary,
        "gate": gate_result(summary),
    }
    text = json.dumps(output, ensure_ascii=False, sort_keys=True)
    if args.save:
        with open(args.save, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(text)
    if args.fail_if_not_better and not output["gate"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
