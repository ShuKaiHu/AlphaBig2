import argparse

import numpy as np
import torch

from ml.baselines import get_baseline
from ml.env import Big2Env
from ml.policy import ActorCriticNet


def select_action(policy, obs, device):
    obs_tensor = torch.from_numpy(obs["obs"]).to(device).unsqueeze(0)
    mask_tensor = torch.from_numpy(obs["action_mask"]).to(device).unsqueeze(0)
    logits, _ = policy(obs_tensor, mask_tensor)
    dist = torch.distributions.Categorical(logits=logits)
    return dist.sample().item()


def play_episode(env, policy, device, mode, max_turns, main_player, baseline_name):
    obs = env.reset()
    done = False
    steps = 0
    reward_vector = None
    baseline = get_baseline(baseline_name) if mode == "vs-baseline" else None

    while not done and steps < max_turns:
        if mode in ("vs-random", "vs-baseline") and obs["player"] != main_player:
            if mode == "vs-random":
                action = get_baseline("random")(obs["action_mask"])
            else:
                action = baseline(obs["action_mask"])
        else:
            action = select_action(policy, obs, device)
        obs, done, info = env.step(action)
        reward_vector = info["reward_vector"]
        steps += 1

    if reward_vector is None:
        reward_vector = np.zeros((4,), dtype=np.float32)
    return reward_vector, steps


def evaluate(args):
    device = torch.device("cpu")
    env = Big2Env()
    policy = ActorCriticNet().to(device)
    payload = torch.load(args.model, map_location=device)
    policy.load_state_dict(payload["model_state"])
    policy.eval()

    wins = 0
    total_rewards = 0.0
    total_steps = 0
    rewards = []
    for _ in range(args.episodes):
        reward_vector, steps = play_episode(
            env,
            policy,
            device,
            args.mode,
            args.max_turns,
            args.main_player,
            args.baseline,
        )
        total_steps += steps
        r_main = float(reward_vector[args.main_player - 1])
        total_rewards += r_main
        rewards.append(r_main)
        if reward_vector[args.main_player - 1] > 0:
            wins += 1

    win_rate = wins / max(args.episodes, 1)
    avg_reward = total_rewards / max(args.episodes, 1)
    avg_steps = total_steps / max(args.episodes, 1)
    rewards_arr = np.array(rewards, dtype=np.float32)
    p25, p50, p75 = np.percentile(rewards_arr, [25, 50, 75])
    hist_bins = np.arange(-40, 41, 5, dtype=np.int32)
    hist_counts, _ = np.histogram(rewards_arr, bins=hist_bins)
    print(
        f"episodes={args.episodes} win_rate={win_rate:.2f} "
        f"avg_reward={avg_reward:.2f} avg_steps={avg_steps:.1f}"
    )
    print(f"reward_p25={p25:.1f} reward_p50={p50:.1f} reward_p75={p75:.1f}")
    print("reward_histogram (bins of 5):")
    for i in range(len(hist_counts)):
        lo = hist_bins[i]
        hi = hist_bins[i + 1]
        print(f"{lo:>3} to {hi:>3}: {hist_counts[i]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument(
        "--mode",
        type=str,
        choices=["selfplay", "vs-random", "vs-baseline"],
        default="vs-baseline",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        choices=["random", "min-index", "max-index"],
        default="random",
    )
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max-turns", type=int, default=500)
    parser.add_argument("--main-player", type=int, default=1)
    evaluate(parser.parse_args())
