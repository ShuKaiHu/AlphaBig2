import argparse
import numpy as np

import big2Game
import enumerateOptions


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


def play_game(random_player):
    game = big2Game.big2Game()
    while not game.gameOver:
        if game.playersGo == random_player:
            action = random_action(game)
        else:
            action = heuristic_action(game)
        if action == enumerateOptions.passInd:
            game.updateGame(-1)
        else:
            opt, n = enumerateOptions.getOptionNC(action)
            game.updateGame(opt, n)
    rewards = [float(r) for r in game.rewards]
    wins = [1 if r > 0 else 0 for r in rewards]
    return wins, rewards


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=1000)
    args = parser.parse_args()

    for rp in range(1, 5):
        wins = [0, 0, 0, 0]
        rewards = [[], [], [], []]
        for _ in range(args.games):
            w, r = play_game(rp)
            for i in range(4):
                wins[i] += w[i]
                rewards[i].append(r[i])
        print(f"random_player: P{rp}  games={args.games}")
        for i in range(4):
            win_rate = wins[i] / args.games
            avg_reward = float(np.mean(rewards[i])) if rewards[i] else 0.0
            print(f"  player{i+1} win_rate: {win_rate:.2f} avg_reward: {avg_reward:.2f}")


if __name__ == "__main__":
    main()
