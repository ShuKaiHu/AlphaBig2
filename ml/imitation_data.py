import argparse
import numpy as np

import big2Game
import enumerateOptions

from ml.mcts import mcts_policy, mcts_search


def generate_dataset(
    num_games,
    mcts_player=1,
    mcts_simulations=50,
    mcts_rollout_steps=100,
    max_turns=300,
    seed=None,
):
    if seed is not None:
        np.random.seed(seed)

    obs_list = []
    mask_list = []
    action_list = []

    for _ in range(num_games):
        game = big2Game.big2Game()
        turn = 0
        while not game.gameOver and turn < max_turns:
            if game.playersGo == mcts_player:
                avail = game.returnAvailableActions()
                action_mask = np.isfinite(big2Game.convertAvailableActions(avail))
                probs = mcts_policy(
                    game,
                    simulations=mcts_simulations,
                    max_rollout_steps=mcts_rollout_steps,
                )
                obs = np.asarray(game.neuralNetworkInputs[game.playersGo], dtype=np.float32)
                obs_list.append(obs)
                mask_list.append(action_mask.astype(np.float32))
                action_list.append(probs)

            # step with heuristic/random for non-MCTS or after labeling
            valid = np.flatnonzero(game.returnAvailableActions() == 1)
            non_pass = valid[valid != enumerateOptions.passInd]
            if non_pass.size > 0:
                act = int(np.random.choice(non_pass))
            elif valid.size > 0:
                act = int(np.random.choice(valid))
            else:
                act = enumerateOptions.passInd

            if act == enumerateOptions.passInd:
                game.updateGame(-1)
            else:
                opt, nC = enumerateOptions.getOptionNC(act)
                game.updateGame(opt, nC)
            turn += 1

    return (
        np.stack(obs_list, axis=0),
        np.stack(mask_list, axis=0),
        np.stack(action_list, axis=0),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument("--mcts-player", type=int, default=1)
    parser.add_argument("--mcts-simulations", type=int, default=50)
    parser.add_argument("--mcts-rollout-steps", type=int, default=100)
    parser.add_argument("--max-turns", type=int, default=300)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", type=str, default="ml/imitation_dataset.npz")
    args = parser.parse_args()

    obs, mask, actions = generate_dataset(
        args.games,
        mcts_player=args.mcts_player,
        mcts_simulations=args.mcts_simulations,
        mcts_rollout_steps=args.mcts_rollout_steps,
        max_turns=args.max_turns,
        seed=args.seed,
    )
    np.savez(args.out, obs=obs, mask=mask, action_probs=actions)
    print(f"saved: {args.out} samples={obs.shape[0]}")


if __name__ == "__main__":
    main()
