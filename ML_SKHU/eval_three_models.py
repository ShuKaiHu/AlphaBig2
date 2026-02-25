import argparse
import numpy as np
import torch

import big2Game
import enumerateOptions

from ML_SKHU.belief import BeliefModel
from ML_SKHU.policy import PolicyModel
from ML_SKHU.value import ValueModel
from ML_SKHU.features import belief_input_dim, encode_policy_input
from ML_SKHU.selfplay import make_policy_fn, make_value_fn
from ML_SKHU.mcts import MCTS


def heuristic_action(game):
    avail = game.returnAvailableActions()
    valid = np.flatnonzero(avail == 1)
    if valid.size == 0:
        return enumerateOptions.passInd
    non_pass = valid[valid != enumerateOptions.passInd]
    if non_pass.size > 0:
        return int(np.min(non_pass))
    return int(enumerateOptions.passInd)


def random_action(game):
    avail = game.returnAvailableActions()
    valid = np.flatnonzero(avail == 1)
    if valid.size == 0:
        return enumerateOptions.passInd
    return int(np.random.choice(valid))


def apply_action(game, action):
    if action == enumerateOptions.passInd:
        game.updateGame(-1)
    else:
        opt, n = enumerateOptions.getOptionNC(action)
        game.updateGame(opt, n)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--belief-ckpt", required=True)
    parser.add_argument("--policy-ckpt", required=True)
    parser.add_argument("--value-ckpt", required=True)
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument("--simulations", type=int, default=50)
    parser.add_argument("--opponent", choices=["heuristic", "random"], default="heuristic")
    parser.add_argument("--belief-mode", choices=["model", "oracle", "uniform"], default="model")
    parser.add_argument("--value-scale", type=float, default=15.0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    b_dim = belief_input_dim()
    dummy = big2Game.big2Game()
    p_dim = int(encode_policy_input(dummy, 1, np.full((52, 3), 1.0 / 3.0, dtype=np.float32)).shape[0])

    belief_model = BeliefModel(b_dim).to(args.device)
    policy_model = PolicyModel(p_dim).to(args.device)
    value_model = ValueModel(p_dim).to(args.device)
    belief_model.load_state_dict(torch.load(args.belief_ckpt, map_location=args.device))
    policy_model.load_state_dict(torch.load(args.policy_ckpt, map_location=args.device))
    value_model.load_state_dict(torch.load(args.value_ckpt, map_location=args.device))
    belief_model.eval()
    policy_model.eval()
    value_model.eval()

    policy_fn = make_policy_fn(
        belief_model, policy_model, device=args.device, belief_mode=args.belief_mode
    )
    value_fn = make_value_fn(
        belief_model, value_model, device=args.device, belief_mode=args.belief_mode
    )
    mcts = MCTS(policy_fn=policy_fn, value_fn=value_fn, n_simulations=args.simulations)

    wins = np.zeros(4, dtype=np.float32)
    rewards = [[] for _ in range(4)]
    for _ in range(args.games):
        game = big2Game.big2Game()
        while not game.gameOver:
            p = game.playersGo
            if p == 1:
                action, _ = mcts.select_action(
                    game,
                    1,
                    temperature=0.0,
                    value_scale=args.value_scale,
                )
            else:
                action = heuristic_action(game) if args.opponent == "heuristic" else random_action(game)
            apply_action(game, action)
        r = [float(x) for x in game.rewards]
        for i in range(4):
            wins[i] += 1.0 if r[i] > 0 else 0.0
            rewards[i].append(r[i])

    print(f"games: {args.games}")
    for i in range(4):
        print(f"player{i+1} win_rate: {wins[i]/args.games:.2f} avg_reward: {np.mean(rewards[i]):.2f}")


if __name__ == "__main__":
    main()
