import csv
import argparse

import torch
import numpy as np

import big2Game
import enumerateOptions

from ML_SKHU.belief import BeliefModel
from ML_SKHU.features import encode_belief_input, belief_targets


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


def _phase(turn, total_turns):
    if total_turns <= 0:
        return "mid"
    third = total_turns // 3
    if turn < third:
        return "early"
    if turn < 2 * third:
        return "mid"
    return "late"


def run_episode(model, max_turns, margin):
    game = big2Game.big2Game()
    rows = []
    turn = 0
    while not game.gameOver and turn < max_turns:
        player = game.playersGo
        belief_in = encode_belief_input(game, 1)
        with torch.no_grad():
            logits = model(torch.from_numpy(belief_in).float().unsqueeze(0))
            probs = torch.softmax(logits, dim=-1).squeeze(0).numpy()
        targets, mask = belief_targets(game, 1)
        valid = mask > 0
        top2 = np.partition(probs, -2, axis=1)
        margin_values = top2[:, -1] - top2[:, -2]
        preds = np.argmax(probs, axis=1)
        if valid.sum() == 0:
            unknown_rate = 0.0
            known_rate = 0.0
            avg_margin = 0.0
            acc_known = 0.0
            known_count = 0
            unknown_count = 0
        else:
            valid_margins = margin_values[valid]
            unknown_rate = float((valid_margins < margin).mean())
            known_rate = 1.0 - unknown_rate
            avg_margin = float(valid_margins.mean())
            known_mask = valid & (margin_values >= margin)
            known_count = int(known_mask.sum())
            unknown_count = int(valid.sum() - known_count)
            if known_count == 0:
                acc_known = 0.0
            else:
                acc_known = float((preds[known_mask] == targets[known_mask]).mean())
        rows.append(
            {
                "turn": turn,
                "player": player,
                "unknown_rate": f"{float(unknown_rate):.4f}",
                "known_rate": f"{float(known_rate):.4f}",
                "avg_margin": f"{float(avg_margin):.4f}",
                "acc_known": f"{float(acc_known):.4f}",
                "known_count": known_count,
                "unknown_count": unknown_count,
            }
        )
        action = random_non_pass(game)
        if action == enumerateOptions.passInd:
            game.updateGame(-1)
        else:
            opt, n = enumerateOptions.getOptionNC(action)
            game.updateGame(opt, n)
        turn += 1
    # Fill phase based on actual total turns in this game.
    total_turns = len(rows)
    for r in rows:
        r["phase"] = _phase(r["turn"], total_turns)
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
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "turn",
                "phase",
                "player",
                "unknown_rate",
                "known_rate",
                "avg_margin",
                "acc_known",
                "known_count",
                "unknown_count",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
