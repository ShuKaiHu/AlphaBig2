import argparse
import numpy as np

import big2Game
import enumerateOptions

from ML_SKHU.features import encode_belief_input, belief_targets
from ML_SKHU.selfplay import _min_play_action


def random_action(game):
    avail = game.returnAvailableActions()
    valid = np.flatnonzero(avail == 1)
    if valid.size == 0:
        return enumerateOptions.passInd
    return int(np.random.choice(valid))


def collect_data(games, opponent="heuristic"):
    obs = []
    masks = []
    actions = []
    values = []
    for _ in range(games):
        game = big2Game.big2Game()
        while not game.gameOver:
            player = game.playersGo
            belief_in = encode_belief_input(game, player)
            mask = np.isfinite(big2Game.convertAvailableActions(game.returnAvailableActions().astype(np.float32))).astype(np.float32)

            if player == 1:
                action = _min_play_action(game) if opponent == "heuristic" else random_action(game)
            else:
                action = _min_play_action(game) if opponent == "heuristic" else random_action(game)

            opt, n = enumerateOptions.getOptionNC(action)
            game.updateGame(opt, n)

            obs.append(belief_in)
            masks.append(mask)
            actions.append(action)

        reward = game.rewards[0]
        for _ in range(len(obs) - len(values)):
            values.append(1.0 if reward > 0 else -1.0)

    return np.stack(obs), np.stack(masks), np.array(actions), np.array(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--opponent", choices=["heuristic", "random"], default="heuristic")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    obs, masks, actions, values = collect_data(args.games, opponent=args.opponent)
    np.savez(args.out, obs=obs, mask=masks, actions=actions, values=values)
    print(f"saved {args.out} samples={obs.shape[0]}")


if __name__ == "__main__":
    main()
