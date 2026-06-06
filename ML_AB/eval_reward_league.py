import argparse
import json
from pathlib import Path

import numpy as np

import big2Game
import enumerateOptions

from ML_AB.agents import ModelAgent, heuristic_action, random_action
from ML_AB.eval import load_model
from ML_AB.live_mcts import LiveBeliefMCTSAgent
from ML_AB.search import ModelMCTS
from ML_AB.state import action_mask, apply_action
from ML_AB.utils import configure_torch_threads, device_from_arg, set_seed


def _build_live_agent(model, device, args):
    return LiveBeliefMCTSAgent(
        model,
        device=device,
        time_limit_sec=args.mcts_seconds,
        determinizations=args.live_determinizations,
        simulations=args.mcts_simulations,
        max_children=args.mcts_max_children,
        value_scale=args.value_scale,
        belief_blend=args.live_belief_blend,
        selection_metric=args.live_selection,
        posterior_particles=args.live_posterior_particles,
        history_likelihood_weight=args.live_history_weight,
        root_warmup_children=args.mcts_root_warmup,
        root_q_min_actions=args.live_root_q_min_actions,
        root_q_min_coverage=args.live_root_q_min_coverage,
        root_q_max_required=args.live_root_q_max_required,
        root_q_min_margin=args.live_root_q_min_margin,
        action_value_fallback_weight=args.live_action_value_fallback_weight,
        seed=args.seed,
    )


def _parse_ckpts(text):
    ckpts = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        if "=" in item:
            name, path = item.split("=", 1)
            ckpts.append((name.strip(), path.strip()))
        else:
            path = item
            ckpts.append((Path(path).stem, path))
    if not ckpts:
        raise ValueError("no checkpoints provided")
    return ckpts


def _model_action(model, device, game, player, args, live_agent=None):
    if args.agent == "live_mcts":
        if int(player) != 1:
            raise ValueError("live_mcts evaluation only supports the model as player 1")
        agent = live_agent if live_agent is not None else _build_live_agent(model, device, args)
        legal = [int(action) for action in np.flatnonzero(action_mask(game) > 0)]
        scores, _policy_logits, _value, _belief, _diagnostics = agent.search(
            game,
            legal_action_indices=legal,
        )
        return int(np.argmax(scores))
    if args.agent == "mcts":
        mcts = ModelMCTS(
            model,
            device=device,
            simulations=args.mcts_simulations,
            move_time_limit_sec=args.mcts_seconds,
            value_scale=args.value_scale,
            max_children=args.mcts_max_children,
            root_warmup_children=args.mcts_root_warmup,
        )
        action, _visits = mcts.search(game, player, temperature=0.0, add_noise=False)
        return int(action)
    agent = ModelAgent(model, device=device, temperature=0.0)
    return int(agent.select_action(game, player))


def _opponent_action(game, opponent):
    if opponent == "heuristic":
        return int(heuristic_action(game))
    if opponent == "random":
        return int(random_action(game))
    raise ValueError(f"unknown opponent: {opponent}")


def _play_one(model, device, opponent, seed, args, live_agent=None):
    set_seed(seed)
    game = big2Game.big2Game()
    turns = 0
    while not game.gameOver and turns < args.max_turns:
        player = int(game.playersGo)
        if player == 1 or opponent == "self":
            action = _model_action(model, device, game, player, args, live_agent=live_agent)
        else:
            action = _opponent_action(game, opponent)
        if int(action) != enumerateOptions.passInd and action < 0:
            action = enumerateOptions.passInd
        apply_action(game, int(action))
        turns += 1
    if game.gameOver:
        rewards = np.asarray(game.rewards, dtype=np.float32)
    else:
        counts = np.asarray([len(game.currentHands[p]) for p in range(1, 5)], dtype=np.float32)
        rewards = -counts
        leader = int(np.argmin(counts))
        rewards[leader] = float(np.sum(counts) - counts[leader])
    return rewards, bool(game.gameOver), turns


def eval_ckpt(name, ckpt, opponents, seeds, device, args):
    model = load_model(ckpt, device)
    live_agent = _build_live_agent(model, device, args) if args.agent == "live_mcts" else None
    rows = []
    all_p1 = []
    completed = 0
    turn_counts = []
    for opponent in opponents:
        rewards = []
        for seed in seeds:
            reward, done, turns = _play_one(model, device, opponent, int(seed), args, live_agent=live_agent)
            rewards.append(reward)
            all_p1.append(float(reward[0]))
            completed += int(done)
            turn_counts.append(int(turns))
        arr = np.stack(rewards)
        rows.append(
            {
                "opponent": opponent,
                "games": int(len(seeds)),
                "p1_avg_reward": float(arr[:, 0].mean()),
                "p1_median_reward": float(np.median(arr[:, 0])),
                "p1_big_loss_rate": float(np.mean(arr[:, 0] <= -10.0)),
                "p1_win_rate": float(np.mean(arr[:, 0] > 0.0)),
                "avg_reward_by_seat": arr.mean(axis=0).round(4).tolist(),
            }
        )
    p1 = np.asarray(all_p1, dtype=np.float32)
    return {
        "name": name,
        "ckpt": ckpt,
        "agent": args.agent,
        "games": int(len(all_p1)),
        "completed_games": int(completed),
        "avg_turns": float(np.mean(turn_counts)) if turn_counts else 0.0,
        "p1_avg_reward": float(p1.mean()) if p1.size else 0.0,
        "p1_median_reward": float(np.median(p1)) if p1.size else 0.0,
        "p1_big_loss_rate": float(np.mean(p1 <= -10.0)) if p1.size else 0.0,
        "p1_win_rate": float(np.mean(p1 > 0.0)) if p1.size else 0.0,
        "by_opponent": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpts", required=True, help="comma-separated name=path entries")
    parser.add_argument("--opponents", default="heuristic,random")
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--seed", type=int, default=260603)
    parser.add_argument("--agent", choices=["model", "mcts", "live_mcts"], default="model")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--value-scale", type=float, default=15.0)
    parser.add_argument("--max-turns", type=int, default=300)
    parser.add_argument("--mcts-simulations", type=int, default=32)
    parser.add_argument("--mcts-seconds", type=float, default=0.0)
    parser.add_argument("--mcts-max-children", type=int, default=64)
    parser.add_argument("--mcts-root-warmup", type=int, default=8)
    parser.add_argument("--live-determinizations", type=int, default=4)
    parser.add_argument("--live-belief-blend", type=float, default=0.2)
    parser.add_argument("--live-posterior-particles", type=int, default=24)
    parser.add_argument("--live-history-weight", type=float, default=1.0)
    parser.add_argument("--live-selection", default="q")
    parser.add_argument("--live-root-q-min-actions", type=int, default=2)
    parser.add_argument("--live-root-q-min-coverage", type=float, default=0.7)
    parser.add_argument("--live-root-q-max-required", type=int, default=8)
    parser.add_argument("--live-root-q-min-margin", type=float, default=0.01)
    parser.add_argument("--live-action-value-fallback-weight", type=float, default=0.0)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    configure_torch_threads()
    device = device_from_arg(args.device)
    ckpts = _parse_ckpts(args.ckpts)
    opponents = [item.strip() for item in str(args.opponents).split(",") if item.strip()]
    seeds = [int(args.seed) + idx for idx in range(int(args.games))]

    results = [eval_ckpt(name, ckpt, opponents, seeds, device, args) for name, ckpt in ckpts]
    baseline = results[0]["p1_avg_reward"]
    for row in results:
        row["delta_vs_baseline"] = float(row["p1_avg_reward"] - baseline)
    summary = {
        "agent": args.agent,
        "opponents": opponents,
        "games_per_opponent": int(args.games),
        "seed_start": int(args.seed),
        "baseline": results[0]["name"],
        "results": results,
    }
    text = json.dumps(summary, sort_keys=True)
    print(text)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(text + "\n")


if __name__ == "__main__":
    main()
