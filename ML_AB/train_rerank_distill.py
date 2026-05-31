import argparse
import json
import os
import time

import numpy as np
import torch

from ML_AB.agents import RerankAgent
from ML_AB.data import Replay, collect_episode
from ML_AB.eval import load_model
from ML_AB.models import checkpoint_payload
from ML_AB.train import train_step
from ML_AB.utils import configure_torch_threads, device_from_arg, set_seed


def _annealed(start, final, step, total):
    total = max(int(total), 1)
    frac = min(max((int(step) - 1) / float(total), 0.0), 1.0)
    return float(start) + (float(final) - float(start)) * frac


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", default="ML_AB/models/big2_transformer_best.pt")
    parser.add_argument("--save", default="ML_AB/models/big2_transformer_rerank_distill.pt")
    parser.add_argument("--metrics", default="ML_AB/runs/rerank_distill_metrics.jsonl")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--updates-per-episode", type=int, default=4)
    parser.add_argument("--buffer-capacity", type=int, default=100000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=260526)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--value-scale", type=float, default=15.0)
    parser.add_argument("--value-weight", type=float, default=0.5)
    parser.add_argument("--belief-weight", type=float, default=0.15)
    parser.add_argument("--max-turns", type=int, default=240)
    parser.add_argument("--control-five-bonus", type=float, default=1.2)
    parser.add_argument("--card-count-bonus", type=float, default=0.12)
    parser.add_argument("--finish-bonus", type=float, default=3.0)
    parser.add_argument("--urgent-opponent-count", type=int, default=3)
    parser.add_argument("--urgent-five-bonus", type=float, default=1.0)
    parser.add_argument("--preserve-five-card-penalty", type=float, default=0.25)
    parser.add_argument("--five-card-margin-weight", type=float, default=0.35)
    parser.add_argument("--five-card-margin-final-weight", type=float, default=0.0)
    parser.add_argument("--five-card-margin-anneal-episodes", type=int, default=300)
    parser.add_argument("--five-card-margin", type=float, default=0.6)
    args = parser.parse_args()

    configure_torch_threads()
    set_seed(args.seed)
    device = device_from_arg(args.device)
    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    os.makedirs(os.path.dirname(args.metrics), exist_ok=True)

    model = load_model(args.init, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    replay = Replay(args.buffer_capacity)
    metrics_file = open(args.metrics, "a", buffering=1)
    t0 = time.time()
    last = {}

    for ep in range(1, args.episodes + 1):
        model.eval()
        margin_weight = _annealed(
            args.five_card_margin_weight,
            args.five_card_margin_final_weight,
            ep,
            args.five_card_margin_anneal_episodes,
        )
        teacher = RerankAgent(
            model,
            device=device,
            temperature=0.0,
            control_five_bonus=args.control_five_bonus,
            card_count_bonus=args.card_count_bonus,
            finish_bonus=args.finish_bonus,
            urgent_opponent_count=args.urgent_opponent_count,
            urgent_five_bonus=args.urgent_five_bonus,
            preserve_five_card_penalty=args.preserve_five_card_penalty,
        )
        samples, rewards = collect_episode(
            policy_mix=("model",),
            model_agent=teacher,
            value_scale=args.value_scale,
            max_turns=args.max_turns,
        )
        replay.add_many(samples)

        if len(replay) >= args.batch_size:
            model.train()
            for _ in range(args.updates_per_episode):
                last = train_step(
                    model,
                    optimizer,
                    replay.sample(args.batch_size),
                    device,
                    value_weight=args.value_weight,
                    belief_weight=args.belief_weight,
                    five_card_margin_weight=margin_weight,
                    five_card_margin=args.five_card_margin,
                )

        row = {
            "episode": ep,
            "buffer": len(replay),
            "samples": len(samples),
            "reward_mean": float(np.mean(rewards)),
            "reward_p1": float(rewards[0]),
            "five_card_margin_weight": float(margin_weight),
            "elapsed_sec": round(time.time() - t0, 3),
            **last,
        }
        metrics_file.write(json.dumps(row, sort_keys=True) + "\n")
        if ep == 1 or ep % 10 == 0:
            print(json.dumps(row, sort_keys=True))

    payload = checkpoint_payload(model, vars(args), {"last": row})
    torch.save(payload, args.save)
    print(f"saved checkpoint: {args.save}")


if __name__ == "__main__":
    main()
