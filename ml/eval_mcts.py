import argparse
import numpy as np
import torch

import big2Game
import enumerateOptions

from ml.mcts import mcts_search
from ml.train_belief import BeliefNet
from ml.policy import PolicyNet


def play_episode(args, belief_model, rollout_policy):
    game = big2Game.big2Game()
    steps = 0
    while not game.gameOver and steps < args.max_turns:
        player = game.playersGo
        if player == args.mcts_player:
            action = mcts_search(
                game,
                simulations=args.simulations,
                max_rollout_steps=args.rollout_steps,
                belief_model=belief_model,
                temperature=args.temperature,
                risk_aversion=args.risk_aversion,
                rollout_policy=rollout_policy,
                policy_model=rollout_policy,
            )
        else:
            valid = np.flatnonzero(game.returnAvailableActions() == 1)
            non_pass = valid[valid != enumerateOptions.passInd]
            if non_pass.size > 0:
                action = int(np.random.choice(non_pass))
            elif valid.size > 0:
                action = int(np.random.choice(valid))
            else:
                action = enumerateOptions.passInd

        if action == enumerateOptions.passInd:
            game.updateGame(-1)
        else:
            opt, nC = enumerateOptions.getOptionNC(action)
            game.updateGame(opt, nC)
        steps += 1
    return game.rewards, steps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--mcts-player", type=int, default=1)
    parser.add_argument("--simulations", type=int, default=100)
    parser.add_argument("--rollout-steps", type=int, default=200)
    parser.add_argument("--max-turns", type=int, default=300)
    parser.add_argument("--belief-model", type=str, default="")
    parser.add_argument("--belief-data", type=str, default="ml/belief_dataset.npz")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--risk-aversion", type=float, default=0.0)
    parser.add_argument("--rollout-model", type=str, default="")
    args = parser.parse_args()

    belief_model = None
    if args.belief_model:
        data = np.load(args.belief_data)
        belief_model = BeliefNet(data["x"].shape[1])
        payload = torch.load(args.belief_model, map_location="cpu")
        belief_model.load_state_dict(payload["model_state"])
        belief_model.eval()
    rollout_policy = None
    if args.rollout_model:
        rollout_policy = PolicyNet()
        payload = torch.load(args.rollout_model, map_location="cpu")
        rollout_policy.load_state_dict(payload["model_state"])
        rollout_policy.eval()

    rewards = []
    steps = []
    wins = 0
    for _ in range(args.episodes):
        reward_vec, step_count = play_episode(args, belief_model, rollout_policy)
        r = float(reward_vec[args.mcts_player - 1])
        rewards.append(r)
        steps.append(step_count)
        if r > 0:
            wins += 1

    rewards = np.array(rewards, dtype=np.float32)
    win_rate = wins / max(args.episodes, 1)
    avg_reward = float(np.mean(rewards))
    p25, p50, p75 = np.percentile(rewards, [25, 50, 75])
    avg_steps = float(np.mean(steps))
    print(
        f"episodes={args.episodes} win_rate={win_rate:.2f} "
        f"avg_reward={avg_reward:.2f} avg_steps={avg_steps:.1f}"
    )
    print(f"reward_p25={p25:.1f} reward_p50={p50:.1f} reward_p75={p75:.1f}")


if __name__ == "__main__":
    main()
