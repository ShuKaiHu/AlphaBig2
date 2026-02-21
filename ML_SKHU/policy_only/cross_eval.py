import argparse
import os

import torch
import numpy as np

import big2Game
import enumerateOptions

from ML_SKHU.policy_value import PolicyValueModel
from ML_SKHU.features import encode_belief_input, encode_full_info_input


def load_policy(path, device="cpu", full_info=False):
    encoder = encode_full_info_input if full_info else encode_belief_input
    sample = encoder(big2Game.big2Game(), 1)
    model = PolicyValueModel(sample.shape[0], hidden_dim=256).to(device)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def heuristic_action(game):
    avail = game.returnAvailableActions()
    valid = np.flatnonzero(avail == 1)
    if valid.size == 0:
        return enumerateOptions.passInd
    non_pass = valid[valid != enumerateOptions.passInd]
    if non_pass.size > 0:
        return int(np.min(non_pass))
    return enumerateOptions.passInd


def action_from_model(model, game, device="cpu", full_info=False):
    encoder = encode_full_info_input if full_info else encode_belief_input
    belief_in = encoder(game, game.playersGo)
    avail = game.returnAvailableActions()
    valid = np.flatnonzero(avail == 1)
    if valid.size == 0:
        return enumerateOptions.passInd
    avail = avail.astype(np.float32)
    mask = torch.tensor(
        np.isfinite(big2Game.convertAvailableActions(avail)).astype(np.float32)
    ).unsqueeze(0).to(device)
    with torch.no_grad():
        logits, _ = model(
            torch.from_numpy(belief_in).float().unsqueeze(0).to(device), mask
        )
    return int(torch.argmax(logits, dim=-1).item())


def run_game(models, device="cpu", full_info=False):
    game = big2Game.big2Game()
    while not game.gameOver:
        player = game.playersGo
        model = models[player - 1]
        if model is None:
            action = heuristic_action(game)
        else:
            action = action_from_model(model, game, device=device, full_info=full_info)
        if action == enumerateOptions.passInd:
            game.updateGame(-1)
        else:
            opt, n = enumerateOptions.getOptionNC(action)
            game.updateGame(opt, n)
    return [float(r) for r in game.rewards]


def format_grouped_table(tables: dict, title: str, labels):
    player_count = len(tables)
    runs = tables[0].shape[0]
    header = "opp\\player"
    col_header = " ".join(f"{label:>10}" for label in labels)
    lines = [title, f"{header:>12} {col_header}"]
    for row_idx in range(runs):
        row_label = labels[row_idx]
        row_vals = []
        for col_idx in range(runs):
            values = [tables[player_idx][row_idx, col_idx] for player_idx in range(player_count)]
            cell = "/".join(f"{val:+.2f}" for val in values)
            row_vals.append(f"{cell:>10}")
        lines.append(f"{row_label:>8} " + " ".join(row_vals))
    return lines


def format_single_table(table: np.ndarray, title: str, labels):
    header = "opp\\player"
    col_header = " ".join(f"{label:>10}" for label in labels)
    lines = [title, f"{header:>12} {col_header}"]
    for row_idx in range(table.shape[0]):
        row_label = labels[row_idx]
        row_vals = [f"{table[row_idx, col_idx]:+10.2f}" for col_idx in range(table.shape[1])]
        lines.append(f"{row_label:>8} " + " ".join(row_vals))
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default="policy_only_runs")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--include-heuristic", action="store_true")
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--full-info", action="store_true")
    args = parser.parse_args()

    model_cache = {}
    if args.include_heuristic:
        run_labels = ["heur"] + [f"run{idx}" for idx in range(1, args.runs + 1)]
    else:
        run_labels = [f"run{idx}" for idx in range(1, args.runs + 1)]
    total_runs = len(run_labels)
    reward_tables = {player: np.zeros((total_runs, total_runs)) for player in range(4)}
    win_tables = {player: np.zeros((total_runs, total_runs)) for player in range(4)}

    for opp_idx, opp_label in enumerate(run_labels):
        if opp_label == "heur":
            opp_model = None
        else:
            opp_path = os.path.join(args.base_dir, opp_label, "policy_only.pt")
            opp_model = model_cache.get(opp_path) or load_policy(
                opp_path, device=args.device, full_info=args.full_info
            )
            model_cache[opp_path] = opp_model
        opp_models = [opp_model] * 3

        for player_run_idx, player_label in enumerate(run_labels):
            if player_label == "heur":
                player_model = None
            else:
                player_path = os.path.join(args.base_dir, player_label, "policy_only.pt")
                player_model = model_cache.get(player_path) or load_policy(
                    player_path, device=args.device, full_info=args.full_info
                )
                model_cache[player_path] = player_model
            models_list = [player_model] + opp_models

            rewards_by_player = [[] for _ in range(4)]
            for _ in range(args.games):
                rewards = run_game(models_list, device=args.device, full_info=args.full_info)
                for player_idx in range(4):
                    rewards_by_player[player_idx].append(rewards[player_idx])

            for p_idx in range(4):
                player_rewards = rewards_by_player[p_idx]
                reward_tables[p_idx][opp_idx, player_run_idx] = np.mean(player_rewards)
                win_tables[p_idx][opp_idx, player_run_idx] = np.mean(
                    [1.0 if r > 0 else 0.0 for r in player_rewards]
                )

    reward_lines = format_grouped_table(
        reward_tables, "reward (rows=opp run, cols=player run groups):", run_labels
    )
    win_lines = format_grouped_table(
        win_tables, "win rate (rows=opp run, cols=player run groups):", run_labels
    )
    p1_reward_lines = format_single_table(
        reward_tables[0], "player1 reward only:", run_labels
    )
    print("\n".join(win_lines))
    print()
    print("\n".join(reward_lines))
    print()
    print("\n".join(p1_reward_lines))


if __name__ == "__main__":
    main()
