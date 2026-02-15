import argparse
import numpy as np
import torch

import big2Game
import enumerateOptions

from ML_SKHU.policy_value import PolicyValueModel
from ML_SKHU.features import encode_belief_input


def load_policy_model(path, device="cpu"):
    sample = encode_belief_input(big2Game.big2Game(), 1)
    model = PolicyValueModel(sample.shape[0], hidden_dim=256).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def random_action(game):
    avail = game.returnAvailableActions()
    valid = np.flatnonzero(avail == 1)
    if valid.size == 0:
        return enumerateOptions.passInd
    return int(np.random.choice(valid))


def heuristic_action(game):
    avail = game.returnAvailableActions()
    valid = np.flatnonzero(avail == 1)
    if valid.size == 0:
        return enumerateOptions.passInd
    non_pass = valid[valid != enumerateOptions.passInd]
    if non_pass.size > 0:
        return int(np.min(non_pass))
    return enumerateOptions.passInd


def model_action(model, game, device="cpu"):
    belief_in = encode_belief_input(game, game.playersGo)
    mask = torch.tensor(
        np.isfinite(
            big2Game.convertAvailableActions(
                game.returnAvailableActions().astype(np.float32)
            )
        ).astype(np.float32)
    ).unsqueeze(0).to(device)
    with torch.no_grad():
        logits, _ = model(
            torch.from_numpy(belief_in).float().unsqueeze(0).to(device), mask
        )
    return int(torch.argmax(logits, dim=-1).item())


def play_game(model, device="cpu", opp_actor="heuristic", opp_model=None):
    model.eval()
    game = big2Game.big2Game()
    wins = [0, 0, 0, 0]
    rewards = [[], [], [], []]
    while not game.gameOver:
        player = game.playersGo
        if player == 1:
            action = model_action(model, game, device=device)
        else:
            if opp_actor == "heuristic":
                action = heuristic_action(game)
            elif opp_actor == "random":
                action = random_action(game)
            elif opp_actor == "model":
                action = model_action(opp_model, game, device=device)
            else:
                raise ValueError(f"unsupported opp_actor {opp_actor}")

        if action == enumerateOptions.passInd:
            game.updateGame(-1)
        else:
            opt, n = enumerateOptions.getOptionNC(action)
            game.updateGame(opt, n)

    reward = [float(r) for r in game.rewards]
    for i in range(4):
        wins[i] = 1 if reward[i] > 0 else 0
        rewards[i].append(reward[i])
    return wins, rewards


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-ckpt", required=True)
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument("--opponent", choices=["heuristic", "random", "model"], default="heuristic")
    parser.add_argument("--opponent-model", default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    sample_input = encode_belief_input(big2Game.big2Game(), 1)
    model = PolicyValueModel(sample_input.shape[0], hidden_dim=256).to(args.device)
    model.load_state_dict(torch.load(args.policy_ckpt, map_location=args.device))

    opp_model = None
    if args.opponent == "model":
        opp_model = load_policy_model(args.opponent_model, device=args.device)

    wins = np.zeros(4)
    rewards = [[] for _ in range(4)]
    for _ in range(args.games):
        w, r = play_game(model, device=args.device, opp_actor=args.opponent, opp_model=opp_model)
        wins += np.array(w)
        for i in range(4):
            rewards[i].append(r[i])

    print(f"games: {args.games}")
    for i in range(4):
        print(f"player{i+1} win_rate: {wins[i]/args.games:.2f} avg_reward: {np.mean(rewards[i]):.2f}")


if __name__ == "__main__":
    main()
