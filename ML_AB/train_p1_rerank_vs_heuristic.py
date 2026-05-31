import argparse
import json
import os
import time

import numpy as np
import torch

import big2Game
import enumerateOptions

from ML_AB.actions import action_cards
from ML_AB.agents import ModelAgent, RerankAgent, heuristic_action
from ML_AB.data import Replay
from ML_AB.eval import load_model
from ML_AB.models import checkpoint_payload
from ML_AB.state import action_mask, apply_action, belief_targets, encode_game, public_belief_prior
from ML_AB.train import train_step
from ML_AB.utils import configure_torch_threads, device_from_arg, set_seed


def _policy_target_from_logits(logits, valid, temperature):
    target = np.zeros_like(logits, dtype=np.float32)
    if float(temperature) <= 0 or valid.size == 0:
        return target
    scores = logits[valid].astype(np.float64)
    scores = scores - np.max(scores)
    probs = np.exp(scores / float(temperature))
    probs /= np.sum(probs)
    target[valid] = probs.astype(np.float32)
    return target


def _annealed(start, final, step, total):
    total = max(int(total), 1)
    frac = min(max((int(step) - 1) / float(total), 0.0), 1.0)
    return float(start) + (float(final) - float(start)) * frac


def _action_diagnostics(mask, action):
    valid = np.flatnonzero(mask > 0)
    five_available = any(len(action_cards(int(a))) == 5 for a in valid)
    non_five_available = any(0 < len(action_cards(int(a))) < 5 for a in valid)
    played_count = len(action_cards(int(action)))
    return {
        "control_five_available": int(five_available),
        "control_non_five_available": int(non_five_available),
        "control_five_played": int(five_available and played_count == 5),
        "control_non_five_when_five_available": int(five_available and 0 < played_count < 5),
    }


def collect_p1_episode(model, device, args, teacher_fraction):
    game = big2Game.big2Game()
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
    learner = ModelAgent(model, device=device, temperature=args.model_temperature)
    use_teacher = bool(np.random.random() < float(teacher_fraction))
    samples = []
    diagnostics = {
        "teacher_episode": int(use_teacher),
        "model_episode": int(not use_teacher),
        "p1_control_turns": 0,
        "control_five_available": 0,
        "control_non_five_available": 0,
        "control_five_played": 0,
        "control_non_five_when_five_available": 0,
    }
    turns = 0
    while not game.gameOver and turns < args.max_turns:
        player = game.playersGo
        if player == 1:
            b_prior = public_belief_prior(game, player)
            encoded = encode_game(game, player, b_prior)
            mask = action_mask(game)
            b_target, b_mask = belief_targets(game, player)
            acting_agent = teacher if use_teacher else learner
            logits, _value = acting_agent.action_logits(game, player)
            valid = np.flatnonzero(mask > 0)
            if use_teacher or args.model_temperature <= 1e-8:
                action = int(valid[np.argmax(logits[valid])]) if valid.size else enumerateOptions.passInd
            else:
                if valid.size:
                    stable = logits[valid] - np.max(logits[valid])
                    probs = np.exp(stable / float(args.model_temperature))
                    probs /= probs.sum()
                    action = int(np.random.choice(valid, p=probs))
                else:
                    action = enumerateOptions.passInd
            sample = {
                "card_feats": encoded["card_feats"],
                "history_feats": encoded["history_feats"],
                "global_feats": encoded["global_feats"],
                "action_mask": mask,
                "action": action,
                "player": player,
                "belief_target": b_target,
                "belief_mask": b_mask,
            }
            if use_teacher and args.distill_temperature > 0:
                sample["policy_target"] = _policy_target_from_logits(
                    logits,
                    valid,
                    args.distill_temperature,
                )
            samples.append(sample)
            if bool(game.control):
                diagnostics["p1_control_turns"] += 1
                for key, value in _action_diagnostics(mask, action).items():
                    diagnostics[key] += value
        else:
            action = heuristic_action(game)
        apply_action(game, action)
        turns += 1

    if game.gameOver:
        rewards = np.array(game.rewards, dtype=np.float32)
    else:
        counts = np.array([len(game.currentHands[p]) for p in range(1, 5)], dtype=np.float32)
        leader = int(np.argmin(counts))
        rewards = -counts
        rewards[leader] = float(np.sum(counts) - counts[leader])

    for sample in samples:
        sample["value_target"] = np.tanh(float(rewards[0]) / float(args.value_scale))
        if not use_teacher:
            sample["policy_gradient"] = True
    return samples, rewards, diagnostics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", default="ML_AB/models/big2_transformer_best.pt")
    parser.add_argument("--save", default="ML_AB/models/big2_transformer_p1_rerank.pt")
    parser.add_argument("--metrics", default="ML_AB/runs/p1_rerank_metrics.jsonl")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--updates-per-episode", type=int, default=8)
    parser.add_argument("--buffer-capacity", type=int, default=100000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=260529)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--value-scale", type=float, default=15.0)
    parser.add_argument("--value-weight", type=float, default=0.15)
    parser.add_argument("--belief-weight", type=float, default=0.02)
    parser.add_argument("--max-turns", type=int, default=160)
    parser.add_argument("--control-five-bonus", type=float, default=1.2)
    parser.add_argument("--card-count-bonus", type=float, default=0.12)
    parser.add_argument("--finish-bonus", type=float, default=3.0)
    parser.add_argument("--urgent-opponent-count", type=int, default=3)
    parser.add_argument("--urgent-five-bonus", type=float, default=1.0)
    parser.add_argument("--preserve-five-card-penalty", type=float, default=0.25)
    parser.add_argument("--distill-temperature", type=float, default=0.35)
    parser.add_argument("--teacher-fraction-start", type=float, default=1.0)
    parser.add_argument("--teacher-fraction-final", type=float, default=0.25)
    parser.add_argument("--teacher-anneal-episodes", type=int, default=500)
    parser.add_argument("--model-temperature", type=float, default=0.2)
    parser.add_argument("--five-card-margin-weight", type=float, default=0.35)
    parser.add_argument("--five-card-margin-final-weight", type=float, default=0.0)
    parser.add_argument("--five-card-margin-anneal-episodes", type=int, default=500)
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
        teacher_fraction = _annealed(
            args.teacher_fraction_start,
            args.teacher_fraction_final,
            ep,
            args.teacher_anneal_episodes,
        )
        margin_weight = _annealed(
            args.five_card_margin_weight,
            args.five_card_margin_final_weight,
            ep,
            args.five_card_margin_anneal_episodes,
        )
        samples, rewards, diagnostics = collect_p1_episode(model, device, args, teacher_fraction)
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
            "reward_p1": float(rewards[0]),
            "teacher_fraction": float(teacher_fraction),
            "five_card_margin_weight": float(margin_weight),
            **diagnostics,
            "elapsed_sec": round(time.time() - t0, 3),
            **last,
        }
        metrics_file.write(json.dumps(row, sort_keys=True) + "\n")
        if ep == 1 or ep % 25 == 0:
            print(json.dumps(row, sort_keys=True))

    payload = checkpoint_payload(model, vars(args), {"last": row})
    torch.save(payload, args.save)
    print(f"saved checkpoint: {args.save}")


if __name__ == "__main__":
    main()
