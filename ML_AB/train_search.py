import argparse
import json
import os
import time

import numpy as np
import torch

from ML_AB.data import Replay, collect_episode
from ML_AB.eval import load_model
from ML_AB.actions import ACTION_DIM
from ML_AB.models import checkpoint_payload
from ML_AB.search import collect_search_episode
from ML_AB.train import train_step
from ML_AB.utils import configure_torch_threads, device_from_arg, set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", default="ML_AB/models/big2_transformer_current.pt")
    parser.add_argument("--save", default="ML_AB/models/big2_transformer_search.pt")
    parser.add_argument("--metrics", default="ML_AB/runs/search_metrics.jsonl")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--simulations", type=int, default=24)
    parser.add_argument("--move-time-limit-sec", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--updates-per-episode", type=int, default=2)
    parser.add_argument("--bootstrap-frac", type=float, default=0.25)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--value-scale", type=float, default=15.0)
    parser.add_argument("--policy-weight", type=float, default=1.0)
    parser.add_argument("--value-weight", type=float, default=0.5)
    parser.add_argument("--q-value-weight", type=float, default=0.25)
    parser.add_argument("--belief-weight", type=float, default=0.2)
    parser.add_argument("--q-head-only", action="store_true")
    parser.add_argument("--max-turns", type=int, default=240)
    args = parser.parse_args()

    configure_torch_threads()
    set_seed(args.seed)
    device = device_from_arg(args.device)
    model = load_model(args.init, device)
    if args.q_head_only:
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith("q_")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    replay = Replay(capacity=100000)

    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    os.makedirs(os.path.dirname(args.metrics), exist_ok=True)
    metrics_file = open(args.metrics, "a", buffering=1)
    t0 = time.time()
    last = {}

    for ep in range(1, args.episodes + 1):
        model.eval()
        if np.random.random() < args.bootstrap_frac:
            samples, rewards = collect_episode(
                policy_mix=("heuristic", "random"),
                model_agent=None,
                value_scale=args.value_scale,
                max_turns=args.max_turns,
            )
            for item in samples:
                target = np.zeros((ACTION_DIM,), dtype=np.float32)
                target[int(item["action"])] = 1.0
                item["policy_target"] = target
            source = "bootstrap"
        else:
            samples, rewards = collect_search_episode(
                model,
                device=device,
                simulations=args.simulations,
                value_scale=args.value_scale,
                max_turns=args.max_turns,
                move_time_limit_sec=args.move_time_limit_sec,
            )
            source = "search"
        replay.add_many(samples)
        if len(replay) >= args.batch_size:
            model.train()
            for _ in range(args.updates_per_episode):
                last = train_step(
                    model,
                    opt,
                    replay.sample(args.batch_size),
                    device,
                    policy_weight=args.policy_weight,
                    value_weight=args.value_weight,
                    q_value_weight=args.q_value_weight,
                    belief_weight=args.belief_weight,
                )

        row = {
            "episode": ep,
            "source": source,
            "buffer": len(replay),
            "samples": len(samples),
            "reward_p1": float(rewards[0]),
            "q_head_only": bool(args.q_head_only),
            "elapsed_sec": round(time.time() - t0, 3),
            **last,
        }
        metrics_file.write(json.dumps(row, sort_keys=True) + "\n")
        if ep == 1 or ep % 5 == 0:
            print(json.dumps(row, sort_keys=True))

    payload = checkpoint_payload(model, vars(args), {"last": row})
    torch.save(payload, args.save)
    print(f"saved checkpoint: {args.save}")


if __name__ == "__main__":
    main()
