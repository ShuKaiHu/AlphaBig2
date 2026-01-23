import argparse
import numpy as np

import big2Game
import enumerateOptions


def _random_action(game):
    avail = game.returnAvailableActions()
    valid = np.flatnonzero(avail == 1)
    if valid.size == 0:
        return enumerateOptions.passInd
    return int(np.random.choice(valid))


def _apply_action(game, action):
    if action == enumerateOptions.passInd:
        game.updateGame(-1)
        return
    opt, n_cards = enumerateOptions.getOptionNC(action)
    game.updateGame(opt, n_cards)


def play_game():
    game = big2Game.big2Game()
    while not game.gameOver:
        action = _random_action(game)
        _apply_action(game, action)
    rewards = [float(r) for r in game.rewards]
    wins = [1 if r > 0 else 0 for r in rewards]
    return wins, rewards


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=50)
    args = parser.parse_args()

    wins = [0, 0, 0, 0]
    rewards = [[], [], [], []]
    for _ in range(args.games):
        w, r = play_game()
        for i in range(4):
            wins[i] += int(w[i])
            rewards[i].append(r[i])

    print(f"games: {args.games}")
    for i in range(4):
        win_rate = wins[i] / args.games
        avg_reward = float(np.mean(rewards[i])) if rewards[i] else 0.0
        print(f"player{i+1} win_rate: {win_rate:.2f} avg_reward: {avg_reward:.2f}")


if __name__ == "__main__":
    main()
