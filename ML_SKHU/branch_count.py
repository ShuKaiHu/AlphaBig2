import argparse
import numpy as np

import big2Game
import enumerateOptions


def _apply_action(game, action):
    if action == enumerateOptions.passInd:
        game.updateGame(-1)
        return
    opt, n_cards = enumerateOptions.getOptionNC(action)
    game.updateGame(opt, n_cards)


def _valid_actions(game):
    avail = game.returnAvailableActions()
    return np.flatnonzero(avail == 1)


def count_branches(game, depth, max_nodes=200000, log_every=100000):
    # returns (total_nodes, total_leafs, truncated)
    nodes = 0
    leafs = 0
    truncated = False

    def dfs(state, d):
        nonlocal nodes, leafs, truncated
        if truncated:
            return
        nodes += 1
        if log_every and nodes % log_every == 0:
            print(f"progress: nodes={nodes} depth={d}")
        if nodes >= max_nodes:
            truncated = True
            return
        if d == 0 or state.gameOver:
            leafs += 1
            return
        actions = _valid_actions(state)
        for action in actions:
            nxt = state.clone()
            _apply_action(nxt, int(action))
            dfs(nxt, d - 1)

    dfs(game, depth)
    return nodes, leafs, truncated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-nodes", type=int, default=200000)
    parser.add_argument("--warmup-moves", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100000)
    args = parser.parse_args()

    np.random.seed(args.seed)
    game = big2Game.big2Game()

    for _ in range(args.warmup_moves):
        if game.gameOver:
            break
        actions = _valid_actions(game)
        if actions.size == 0:
            break
        action = int(np.random.choice(actions))
        _apply_action(game, action)

    actions = _valid_actions(game)
    print(f"initial legal actions: {len(actions)}")

    nodes, leafs, truncated = count_branches(
        game, args.depth, max_nodes=args.max_nodes, log_every=args.log_every
    )
    print(f"depth={args.depth} total_nodes={nodes} total_leafs={leafs} truncated={truncated}")


if __name__ == "__main__":
    main()
