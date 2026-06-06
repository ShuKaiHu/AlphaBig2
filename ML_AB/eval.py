import argparse
import json

import numpy as np
import torch

import big2Game

from ML_AB.actions import ACTION_FEAT_DIM
from ML_AB.agents import ModelAgent, RerankAgent, heuristic_action, random_action
from ML_AB.models import Big2TransformerNet
from ML_AB.state import apply_action
from ML_AB.utils import configure_torch_threads, device_from_arg, set_seed


def load_model(path, device):
    payload = torch.load(path, map_location=device)
    cfg = payload.get("config", {})
    state = payload["model_state"] if "model_state" in payload else payload
    d_model = int(cfg.get("d_model", state["card_proj.weight"].shape[0]))
    if "history_pos" in state:
        history_len = int(state["history_pos"].shape[1])
    else:
        history_len = int(payload.get("history_len", cfg.get("history_len", 196)))
    layers = cfg.get("layers")
    if layers is None:
        layer_ids = set()
        prefix = "encoder.layers."
        for key in state:
            if key.startswith(prefix):
                layer_ids.add(int(key[len(prefix) :].split(".", 1)[0]))
        layers = max(layer_ids) + 1 if layer_ids else 1
    heads = int(cfg.get("heads", 4 if d_model % 4 == 0 else 1))
    model = Big2TransformerNet(
        d_model=d_model,
        nhead=heads,
        num_layers=int(layers),
        dropout=0.0,
        action_feat_dim=ACTION_FEAT_DIM,
        max_history_len=history_len,
    ).to(device)
    state = _adapt_action_feature_state(state, model.state_dict())
    target_keys = set(model.state_dict())
    state_keys = set(state)
    strict = ("history_pos" in state) and target_keys.issubset(state_keys)
    model.load_state_dict(state, strict=strict)
    model.has_trained_action_value = all(
        key in state_keys
        for key in (
            "q_state_proj.weight",
            "q_action_proj.0.weight",
            "q_head.1.weight",
        )
    ) and _checkpoint_trained_action_value(payload)
    model.eval()
    return model


def _truthy_q_value_flag(row):
    if not isinstance(row, dict):
        return False
    if bool(row.get("q_head_only")):
        return True
    try:
        return float(row.get("q_value_weight", 0.0)) > 0.0
    except (TypeError, ValueError):
        return False


def _checkpoint_trained_action_value(payload):
    if not isinstance(payload, dict):
        return False
    if _truthy_q_value_flag(payload.get("config")):
        return True
    metrics = payload.get("metrics")
    if _truthy_q_value_flag(metrics):
        return True
    if isinstance(metrics, dict) and _truthy_q_value_flag(metrics.get("last")):
        return True
    return False


def _adapt_action_feature_state(state, target_state):
    state = dict(state)
    for key in ("action_proj.0.weight", "action_bias.weight"):
        if key not in state or key not in target_state:
            continue
        src = state[key]
        dst = target_state[key]
        if src.shape == dst.shape:
            continue
        if src.ndim != 2 or dst.ndim != 2 or src.shape[0] != dst.shape[0]:
            continue
        adapted = dst.new_zeros(dst.shape)
        width = min(src.shape[1], dst.shape[1])
        adapted[:, :width] = src[:, :width]
        state[key] = adapted
    return state


def _make_agent(model, device, args):
    agent_name = args.agent
    if agent_name == "model":
        return ModelAgent(model, device=device, temperature=0.0)
    if agent_name == "rerank":
        return RerankAgent(
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
    raise ValueError(f"unknown agent: {agent_name}")


def eval_games(model, games, opponent, device, args):
    agent = _make_agent(model, device, args)
    rewards = []
    wins = np.zeros((4,), dtype=np.float32)
    big_losses = np.zeros((4,), dtype=np.float32)
    for _ in range(games):
        game = big2Game.big2Game()
        while not game.gameOver:
            p = game.playersGo
            if p == 1:
                action = agent.select_action(game, p)
            elif opponent == "heuristic":
                action = heuristic_action(game)
            elif opponent == "random":
                action = random_action(game)
            else:
                action = agent.select_action(game, p)
            apply_action(game, action)
        r = np.array(game.rewards, dtype=np.float32)
        rewards.append(r)
        wins += (r > 0).astype(np.float32)
        big_losses += (r <= -10).astype(np.float32)
    arr = np.stack(rewards)
    return {
        "games": games,
        "agent": args.agent,
        "opponent": opponent,
        "avg_reward": arr.mean(axis=0).round(4).tolist(),
        "median_reward": np.median(arr, axis=0).round(4).tolist(),
        "p1_big_loss_rate": float(big_losses[0] / float(games)),
        "win_rate": (wins / float(games)).round(4).tolist(),
        "p1_avg_reward": float(arr[:, 0].mean()),
        "p1_win_rate": float(wins[0] / float(games)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--opponent", choices=["heuristic", "random", "self"], default="heuristic")
    parser.add_argument("--agent", choices=["model", "rerank"], default="model")
    parser.add_argument("--control-five-bonus", type=float, default=1.2)
    parser.add_argument("--card-count-bonus", type=float, default=0.12)
    parser.add_argument("--finish-bonus", type=float, default=3.0)
    parser.add_argument("--urgent-opponent-count", type=int, default=3)
    parser.add_argument("--urgent-five-bonus", type=float, default=1.0)
    parser.add_argument("--preserve-five-card-penalty", type=float, default=0.25)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    configure_torch_threads()
    set_seed(args.seed)
    device = device_from_arg(args.device)
    model = load_model(args.ckpt, device)
    print(json.dumps(eval_games(model, args.games, args.opponent, device, args), sort_keys=True))


if __name__ == "__main__":
    main()
