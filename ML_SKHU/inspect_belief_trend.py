import csv
import argparse

import torch
import numpy as np

import big2Game
import enumerateOptions

from ML_SKHU.belief import BeliefModel
from ML_SKHU.features import encode_belief_input


def random_non_pass(game):
    avail = game.returnAvailableActions()
    valid = np.flatnonzero(avail == 1)
    non_pass = valid[valid != enumerateOptions.passInd]
    if non_pass.size > 0:
        return int(non_pass[0])
    return enumerateOptions.passInd


def load_belief(path):
    b_input_dim = encode_belief_input(big2Game.big2Game(), 1).shape[0]
    model = BeliefModel(b_input_dim)
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def run_episode(model, max_turns, margin):
    game = big2Game.big2Game()
    rows = []
    turn = 0
    while not game.gameOver and turn < max_turns:
        player = game.playersGo
        belief_in = encode_belief_input(game, player)
        with torch.no_grad():
            logits = model(torch.from_numpy(belief_in).float().unsqueeze(0))
            probs = torch.softmax(logits, dim=-1).squeeze(0).numpy()
        top_idx = np.argmax(probs, axis=1)
        top2 = np.partition(probs, -2, axis=1)
        margin_values = top2[:, -1] - top2[:, -2]
        unknown_rate = (margin_values < margin).mean()
        known_rate = (top_idx != 3).sum() / 52
        avg_margin = margin_values.mean()
        rows.append(
            {
                "turn": turn,
                "player": player,
                "unknown_rate": unknown_rate,
                "known_rate": known_rate,
                "avg_margin": avg_margin,
            }
        )
        action = random_non_pass(game)
        if action == enumerateOptions.passInd:
            game.updateGame(-1)
        else:
            opt, n = enumerateOptions.getOptionNC(action)
            game.updateGame(opt, n)
        turn += 1
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--belief-ckpt", required=True)
    parser.add_argument("--max-turns", type=int, default=100)
    parser.add_argument("--margin", type=float, default=0.05)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    model = load_belief(args.belief_ckpt)
    rows = run_episode(model, args.max_turns, args.margin)

    with open(args.out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["turn", "player", "unknown_rate", "known_rate", "avg_margin"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
