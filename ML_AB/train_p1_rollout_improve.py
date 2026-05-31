import argparse
import json
import os
import time

import numpy as np
import torch

import big2Game

from ML_AB.actions import ACTION_DIM, action_cards
from ML_AB.agents import ModelAgent, heuristic_action, random_action
from ML_AB.data import Replay
from ML_AB.eval import load_model
from ML_AB.models import checkpoint_payload
from ML_AB.state import action_mask, apply_action, belief_targets, encode_game, public_belief_prior
from ML_AB.train import train_step
from ML_AB.utils import configure_torch_threads, device_from_arg, set_seed


def _unfinished_rewards(game):
    counts = np.array([len(game.currentHands[p]) for p in range(1, 5)], dtype=np.float32)
    leader = int(np.argmin(counts))
    rewards = -counts
    rewards[leader] = float(np.sum(counts) - counts[leader])
    return rewards


def _sample_model_action(agent, game, player, temperature):
    logits, _value = agent.action_logits(game, player)
    valid = np.flatnonzero(action_mask(game) > 0)
    if valid.size == 0:
        import enumerateOptions

        return enumerateOptions.passInd
    if float(temperature) <= 1e-8:
        return int(valid[np.argmax(logits[valid])])
    stable = logits[valid] - np.max(logits[valid])
    probs = np.exp(stable / float(temperature))
    probs /= np.sum(probs)
    return int(np.random.choice(valid, p=probs))


def _rollout_value(game, first_action, agent, args):
    rewards = []
    for _ in range(args.rollouts_per_action):
        sim = game.clone()
        apply_action(sim, int(first_action))
        turns = 0
        while not sim.gameOver and turns < args.rollout_max_turns:
            player = sim.playersGo
            if player == 1:
                if args.rollout_p1_policy == "model":
                    action = _sample_model_action(agent, sim, player, args.rollout_model_temperature)
                elif args.rollout_p1_policy == "random":
                    action = random_action(sim)
                else:
                    action = heuristic_action(sim)
            elif np.random.random() < args.rollout_opponent_random_frac:
                action = random_action(sim)
            else:
                action = heuristic_action(sim)
            apply_action(sim, int(action))
            turns += 1
        final_rewards = np.array(sim.rewards, dtype=np.float32) if sim.gameOver else _unfinished_rewards(sim)
        rewards.append(float(np.tanh(float(final_rewards[0]) / float(args.value_scale))))
    return float(np.mean(rewards))


def _candidate_prior(logits, candidates, temperature):
    scores = logits[candidates].astype(np.float64)
    scores = scores - np.max(scores)
    probs = np.exp(scores / max(float(temperature), 1e-8))
    probs /= np.sum(probs)
    return probs


def _candidate_actions(game, logits, valid, args):
    ranked = valid[np.argsort(-logits[valid])]
    selected = list(ranked[: min(args.rollout_topk, ranked.size)])

    hand_count = len(game.currentHands[1])
    if 0 < args.include_all_actions_hand_count and hand_count <= args.include_all_actions_hand_count:
        selected = list(ranked)

    if args.include_all_five_candidates:
        seen = set(int(action) for action in selected)
        for action in ranked:
            if len(action_cards(int(action))) == 5 and int(action) not in seen:
                selected.append(int(action))
                seen.add(int(action))

    if args.max_rollout_candidates > 0 and len(selected) > args.max_rollout_candidates:
        keep = set(int(action) for action in selected[: args.max_rollout_candidates])
        if 0 < args.include_all_actions_hand_count and hand_count <= args.include_all_actions_hand_count:
            keep = set(int(action) for action in selected)
        selected = [int(action) for action in selected if int(action) in keep]

    return np.array(selected, dtype=np.int64)


def _rollout_policy_target(game, agent, args):
    logits, _value = agent.action_logits(game, 1)
    mask = action_mask(game)
    valid = np.flatnonzero(mask > 0)
    if valid.size == 0:
        target = np.zeros((ACTION_DIM,), dtype=np.float32)
        return 0, target, {}

    candidates = _candidate_actions(game, logits, valid, args)
    values = np.array([_rollout_value(game, int(action), agent, args) for action in candidates], dtype=np.float64)
    prior_probs = _candidate_prior(logits, candidates, args.prior_temperature)
    stable = values - np.max(values)
    rollout_probs = np.exp(stable / max(float(args.rollout_target_temperature), 1e-8))
    rollout_probs /= np.sum(rollout_probs)

    target = np.zeros((ACTION_DIM,), dtype=np.float32)
    if args.pairwise_five_non_five_target:
        five_idx = [i for i, action in enumerate(candidates) if len(action_cards(int(action))) == 5]
        non_five_idx = [i for i, action in enumerate(candidates) if 0 < len(action_cards(int(action))) < 5]
        if five_idx and non_five_idx:
            best_five = max(five_idx, key=lambda i: values[i])
            best_non_five = max(non_five_idx, key=lambda i: values[i])
            pair = np.array([best_five, best_non_five], dtype=np.int64)
            pair_values = values[pair]
            pair_stable = pair_values - np.max(pair_values)
            pair_probs = np.exp(pair_stable / max(float(args.rollout_target_temperature), 1e-8))
            pair_probs /= np.sum(pair_probs)
            mixed_probs = np.zeros_like(rollout_probs)
            mixed_probs[pair] = pair_probs
        else:
            mixed_probs = rollout_probs
    else:
        mixed_probs = rollout_probs

    if args.prior_blend > 0:
        mixed_probs = (
            (1.0 - float(args.prior_blend)) * mixed_probs
            + float(args.prior_blend) * prior_probs
        )
        mixed_probs /= np.sum(mixed_probs)

    target[candidates] = mixed_probs.astype(np.float32)
    action = int(candidates[int(np.argmax(values))])

    five_candidates = [int(a) for a in candidates if len(action_cards(int(a))) == 5]
    non_five_candidates = [int(a) for a in candidates if 0 < len(action_cards(int(a))) < 5]
    best_five_value = None
    best_non_five_value = None
    if five_candidates:
        best_five_value = float(
            max(values[i] for i, a in enumerate(candidates) if int(a) in five_candidates)
        )
    if non_five_candidates:
        best_non_five_value = float(
            max(values[i] for i, a in enumerate(candidates) if int(a) in non_five_candidates)
        )
    diag = {
        "rollout_candidates": int(len(candidates)),
        "rollout_all_endgame_candidates": int(
            0 < args.include_all_actions_hand_count
            and len(game.currentHands[1]) <= args.include_all_actions_hand_count
        ),
        "rollout_best_value": float(np.max(values)),
        "rollout_worst_value": float(np.min(values)),
        "rollout_value_spread": float(np.max(values) - np.min(values)),
        "rollout_best_five_value": best_five_value,
        "rollout_best_non_five_value": best_non_five_value,
        "rollout_five_value_minus_non_five": (
            None
            if best_five_value is None or best_non_five_value is None
            else float(best_five_value - best_non_five_value)
        ),
        "rollout_five_candidates": int(len(five_candidates)),
        "rollout_non_five_candidates": int(len(non_five_candidates)),
        "rollout_chose_five": int(len(action_cards(action)) == 5),
        "rollout_chose_non_five_with_five_candidate": int(bool(five_candidates) and 0 < len(action_cards(action)) < 5),
    }
    return action, target, diag


def collect_p1_rollout_episode(model, device, args):
    game = big2Game.big2Game()
    agent = ModelAgent(model, device=device, temperature=0.0)
    samples = []
    totals = {
        "p1_turns": 0,
        "rollout_states": 0,
        "rollout_all_endgame_candidate_states": 0,
        "rollout_five_candidates": 0,
        "rollout_non_five_candidates": 0,
        "rollout_chose_five": 0,
        "rollout_chose_non_five_with_five_candidate": 0,
        "rollout_pairwise_states": 0,
        "rollout_five_better_states": 0,
        "rollout_non_five_better_states": 0,
        "rollout_five_value_margin_sum": 0.0,
        "rollout_value_spread_sum": 0.0,
    }
    turns = 0
    while not game.gameOver and turns < args.max_turns:
        player = game.playersGo
        if player == 1:
            totals["p1_turns"] += 1
            b_prior = public_belief_prior(game, player)
            encoded = encode_game(game, player, b_prior)
            mask = action_mask(game)
            b_target, b_mask = belief_targets(game, player)
            action, target, diag = _rollout_policy_target(game, agent, args)
            samples.append(
                {
                    "card_feats": encoded["card_feats"],
                    "history_feats": encoded["history_feats"],
                    "global_feats": encoded["global_feats"],
                    "action_mask": mask,
                    "action": int(action),
                    "policy_target": target,
                    "player": player,
                    "belief_target": b_target,
                    "belief_mask": b_mask,
                }
            )
            if diag:
                totals["rollout_states"] += 1
                for key in (
                    "rollout_five_candidates",
                    "rollout_non_five_candidates",
                    "rollout_chose_five",
                    "rollout_chose_non_five_with_five_candidate",
                    "rollout_all_endgame_candidates",
                ):
                    if key == "rollout_all_endgame_candidates":
                        totals["rollout_all_endgame_candidate_states"] += int(diag[key])
                    else:
                        totals[key] += int(diag[key])
                margin = diag["rollout_five_value_minus_non_five"]
                if margin is not None:
                    totals["rollout_pairwise_states"] += 1
                    totals["rollout_five_value_margin_sum"] += float(margin)
                    if margin > 0:
                        totals["rollout_five_better_states"] += 1
                    elif margin < 0:
                        totals["rollout_non_five_better_states"] += 1
                totals["rollout_value_spread_sum"] += float(diag["rollout_value_spread"])
        elif np.random.random() < args.opponent_random_frac:
            action = random_action(game)
        else:
            action = heuristic_action(game)
        apply_action(game, int(action))
        turns += 1

    rewards = np.array(game.rewards, dtype=np.float32) if game.gameOver else _unfinished_rewards(game)
    for sample in samples:
        sample["value_target"] = np.tanh(float(rewards[0]) / float(args.value_scale))
    return samples, rewards, totals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", default="ML_AB/models/big2_transformer_best.pt")
    parser.add_argument("--save", default="ML_AB/models/candidate_p1_rollout_improve.pt")
    parser.add_argument("--metrics", default="ML_AB/runs/p1_rollout_improve.jsonl")
    parser.add_argument("--episodes", type=int, default=240)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--updates-per-episode", type=int, default=8)
    parser.add_argument("--buffer-capacity", type=int, default=120000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=260530)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--value-scale", type=float, default=15.0)
    parser.add_argument("--value-weight", type=float, default=0.35)
    parser.add_argument("--belief-weight", type=float, default=0.02)
    parser.add_argument("--max-turns", type=int, default=180)
    parser.add_argument("--rollout-topk", type=int, default=8)
    parser.add_argument("--include-all-actions-hand-count", type=int, default=3)
    parser.add_argument("--include-all-five-candidates", action="store_true")
    parser.add_argument("--max-rollout-candidates", type=int, default=32)
    parser.add_argument("--rollouts-per-action", type=int, default=2)
    parser.add_argument("--rollout-max-turns", type=int, default=120)
    parser.add_argument("--rollout-model-temperature", type=float, default=0.12)
    parser.add_argument("--rollout-p1-policy", choices=["model", "heuristic", "random"], default="model")
    parser.add_argument("--rollout-opponent-random-frac", type=float, default=0.0)
    parser.add_argument("--rollout-target-temperature", type=float, default=0.35)
    parser.add_argument("--prior-temperature", type=float, default=0.8)
    parser.add_argument("--prior-blend", type=float, default=0.35)
    parser.add_argument("--pairwise-five-non-five-target", action="store_true")
    parser.add_argument("--opponent-random-frac", type=float, default=0.0)
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
        samples, rewards, diag = collect_p1_rollout_episode(model, device, args)
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
                )

        rollout_states = max(int(diag["rollout_states"]), 1)
        row = {
            "episode": ep,
            "buffer": len(replay),
            "samples": len(samples),
            "reward_p1": float(rewards[0]),
            "rollout_avg_value_spread": float(diag["rollout_value_spread_sum"] / rollout_states),
            "rollout_avg_five_value_margin": float(
                diag["rollout_five_value_margin_sum"] / max(int(diag["rollout_pairwise_states"]), 1)
            ),
            **diag,
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
