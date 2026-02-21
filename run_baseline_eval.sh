#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
GAMES="${GAMES:-200}"

PASS_ONLY="${PASS_ONLY:-0}"

"$PYTHON_BIN" - <<'PY'
import numpy as np
import big2Game, enumerateOptions

import os
GAMES = int(os.environ.get("GAMES", "200"))
PASS_ONLY = os.environ.get("PASS_ONLY", "0") == "1"

def heuristic_action(game):
    avail = game.returnAvailableActions()
    valid = np.flatnonzero(avail == 1)
    if valid.size == 0:
        return enumerateOptions.passInd
    non_pass = valid[valid != enumerateOptions.passInd]
    return int(np.min(non_pass)) if non_pass.size > 0 else enumerateOptions.passInd

def random_action(game):
    avail = game.returnAvailableActions()
    valid = np.flatnonzero(avail == 1)
    if valid.size == 0:
        return enumerateOptions.passInd
    return int(np.random.choice(valid))

def pass_only_action(game):
    if game.mustPlayClub3 and game.playersGo == game.club3Player:
        avail = game.returnAvailableActions()
        valid = np.flatnonzero(avail == 1)
        for action in valid:
            opt, n = enumerateOptions.getOptionNC(int(action))
            if n == 1 and game.currentHands[game.playersGo][opt] == 1:
                return int(action)
    return enumerateOptions.passInd

def play_game(player1_actor="heuristic"):
    g = big2Game.big2Game()
    while not g.gameOver:
        if g.playersGo == 1:
            if player1_actor == "random":
                action = random_action(g)
            elif player1_actor == "pass_only":
                action = pass_only_action(g)
            else:
                action = heuristic_action(g)
        else:
            action = heuristic_action(g)
        if action == enumerateOptions.passInd:
            g.updateGame(-1)
        else:
            opt, n = enumerateOptions.getOptionNC(action)
            g.updateGame(opt, n)
    return float(g.rewards[0])

rewards_hh = [play_game("heuristic") for _ in range(GAMES)]
rewards_rh = [play_game("random") for _ in range(GAMES)]
rewards_ph = [play_game("pass_only") for _ in range(GAMES)] if PASS_ONLY else None

print(f"games: {GAMES}")
print(f"heur vs heur avg_reward: {np.mean(rewards_hh):.4f}")
print(f"random vs heur avg_reward: {np.mean(rewards_rh):.4f}")
if PASS_ONLY:
    print(f"pass-only vs heur avg_reward: {np.mean(rewards_ph):.4f}")
PY
