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
    mask = torch.tensor(np.isfinite(big2Game.convertAvailableActions(avail)).astype(np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        logits, _ = model(torch.from_numpy(belief_in).float().unsqueeze(0).to(device), mask)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default="policy_only_runs")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    model_cache = {}
    table = np.zeros((args.runs, args.runs))
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
            rewards = []
            for _ in range(args.games):
                reward = run_game(models_list, device=args.device)[0]
                rewards.append(reward)
            table[opp_run - 1, player_run - 1] = np.mean(rewards)

    print("average player1 reward (rows=opponent model run, cols=player1 model run):")
    for row in table:
        print(" ".join(f"{val:.2f}" for val in row))


if __name__ == "__main__":
    main()
