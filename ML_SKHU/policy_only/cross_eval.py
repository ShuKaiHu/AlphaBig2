import argparse
import os

import torch
import numpy as np

import big2Game
import enumerateOptions

from ML_SKHU.policy_value import PolicyValueModel
from ML_SKHU.features import encode_belief_input


def load_policy(path, device="cpu"):
    sample = encode_belief_input(big2Game.big2Game(), 1)
    model = PolicyValueModel(sample.shape[0], hidden_dim=256).to(device)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def action_from_model(model, game, device="cpu"):
    belief_in = encode_belief_input(game, game.playersGo)
    avail = game.returnAvailableActions().astype(np.float32)
    mask = torch.tensor(
        np.isfinite(big2Game.convertAvailableActions(avail)).astype(np.float32)
    ).unsqueeze(0).to(device)
    with torch.no_grad():
        logits, _ = model(
            torch.from_numpy(belief_in).float().unsqueeze(0).to(device), mask
        )
    return int(torch.argmax(logits, dim=-1).item())


def run_game(models, device="cpu"):
    game = big2Game.big2Game()
    while not game.gameOver:
        player = game.playersGo
        model = models[player - 1]
        action = action_from_model(model, game, device=device)
        if action == enumerateOptions.passInd:
            game.updateGame(-1)
        else:
            opt, n = enumerateOptions.getOptionNC(action)
            game.updateGame(opt, n)
    return [float(r) for r in game.rewards]


def format_grouped_table(tables: dict, title: str):
    player_count = len(tables)
    runs = tables[0].shape[0]
    header = "opp\\player"
    col_header = " ".join(f"run{col+1:>10}" for col in range(runs))
    lines = [title, f"{header:>12} {col_header}"]
    for row_idx in range(runs):
        row_vals = []
        for col_idx in range(runs):
            values = [tables[player_idx][row_idx, col_idx] for player_idx in range(player_count)]
            cell = "/".join(f"{val:+.2f}" for val in values)
            row_vals.append(f"{cell:>10}")
        lines.append(f"run{row_idx+1:>8} " + " ".join(row_vals))
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default="policy_only_runs")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    model_cache = {}
    reward_tables = {player: np.zeros((args.runs, args.runs)) for player in range(4)}
    win_tables = {player: np.zeros((args.runs, args.runs)) for player in range(4)}

    for opp_run in range(1, args.runs + 1):
        opp_path = os.path.join(args.base_dir, f"run{opp_run}", "policy_only.pt")
        opp_model = model_cache.get(opp_path) or load_policy(opp_path, device=args.device)
        model_cache[opp_path] = opp_model
        opp_models = [opp_model] * 3

        for player_run in range(1, args.runs + 1):
            player_path = os.path.join(args.base_dir, f"run{player_run}", "policy_only.pt")
            player_model = model_cache.get(player_path) or load_policy(player_path, device=args.device)
            model_cache[player_path] = player_model
            models_list = [player_model] + opp_models

            rewards_by_player = [[] for _ in range(4)]
            for _ in range(args.games):
                rewards = run_game(models_list, device=args.device)
                for player_idx in range(4):
                    rewards_by_player[player_idx].append(rewards[player_idx])

            for player_idx in range(4):
                player_rewards = rewards_by_player[player_idx]
                reward_tables[player_idx][opp_run - 1, player_run - 1] = np.mean(player_rewards)
                win_tables[player_idx][opp_run - 1, player_run - 1] = np.mean(
                    [1.0 if r > 0 else 0.0 for r in player_rewards]
                )

    reward_lines = format_grouped_table(
        reward_tables, "reward (rows=opp run, cols=player run groups):"
    )
    win_lines = format_grouped_table(
        win_tables, "win rate (rows=opp run, cols=player run groups):"
    )
    print("\n".join(win_lines))
    print()
    print("\n".join(reward_lines))


if __name__ == "__main__":
    main()
