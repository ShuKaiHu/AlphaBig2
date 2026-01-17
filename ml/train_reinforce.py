import argparse
import time

import numpy as np
import torch
from torch import optim

from ml.env import Big2Env
from ml.policy import PolicyNet


def run_episode(env, policy, device, max_turns):
    obs = env.reset()
    done = False
    steps = 0
    traj = []
    reward_vector = None

    while not done and steps < max_turns:
        obs_tensor = torch.from_numpy(obs["obs"]).to(device).unsqueeze(0)
        mask_tensor = torch.from_numpy(obs["action_mask"]).to(device).unsqueeze(0)
        logits = policy(obs_tensor, mask_tensor)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        logprob = dist.log_prob(action)
        traj.append((logprob, obs["player"]))
        obs, done, info = env.step(action.item())
        reward_vector = info["reward_vector"]
        steps += 1

    if reward_vector is None:
        reward_vector = np.zeros((4,), dtype=np.float32)
    return traj, reward_vector, steps


def train(args):
    device = torch.device("cpu")
    env = Big2Env()
    policy = PolicyNet().to(device)
    optimizer = optim.Adam(policy.parameters(), lr=args.lr)

    start = time.time()
    for episode in range(1, args.episodes + 1):
        traj, reward_vector, steps = run_episode(env, policy, device, args.max_turns)
        loss = 0.0
        for logprob, player in traj:
            reward = float(reward_vector[player - 1])
            loss = loss - logprob * reward
        loss = loss / max(len(traj), 1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if episode % args.log_every == 0:
            avg_reward = float(np.mean(reward_vector))
            print(
                f"ep={episode} steps={steps} loss={loss.item():.4f} "
                f"avg_reward={avg_reward:.2f}"
            )

    elapsed = time.time() - start
    print(f"done in {elapsed:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-turns", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=10)
    train(parser.parse_args())
