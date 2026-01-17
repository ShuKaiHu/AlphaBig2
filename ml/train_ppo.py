import argparse
import os
import time

import numpy as np
import torch
from torch import optim

from ml.env import Big2Env
from ml.policy import ActorCriticNet
from big2Game import vectorizedBig2Games


def select_action(policy, obs, device):
    obs_tensor = torch.from_numpy(obs["obs"]).to(device).unsqueeze(0)
    mask_tensor = torch.from_numpy(obs["action_mask"]).to(device).unsqueeze(0)
    logits, value = policy(obs_tensor, mask_tensor)
    dist = torch.distributions.Categorical(logits=logits)
    action = dist.sample()
    logprob = dist.log_prob(action)
    return action.item(), logprob.squeeze(0), value.squeeze(0)


def compute_gae(rewards, values, gamma, lam):
    values = values + [0.0]
    advantages = []
    gae = 0.0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
    returns = [a + v for a, v in zip(advantages, values[:-1])]
    return advantages, returns


def _format_state(player, state, avail):
    obs = np.asarray(state, dtype=np.float32).reshape(-1)
    avail = np.asarray(avail, dtype=np.float32).reshape(-1)
    action_mask = np.isfinite(avail)
    return {
        "player": int(player),
        "obs": obs,
        "action_mask": action_mask,
    }


def _random_action(mask):
    valid = np.flatnonzero(mask)
    if valid.size == 0:
        return 0
    return int(np.random.choice(valid))


def run_episode(env, policy, device, max_turns):
    obs = env.reset()
    done = False
    steps = 0
    trajectory = []
    reward_vector = None

    while not done and steps < max_turns:
        action, logprob, value = select_action(policy, obs, device)
        trajectory.append(
            {
                "obs": obs["obs"].copy(),
                "mask": obs["action_mask"].copy(),
                "action": action,
                "logprob": logprob.detach().cpu().item(),
                "value": value.detach().cpu().item(),
                "player": obs["player"],
            }
        )
        obs, done, info = env.step(action)
        reward_vector = info["reward_vector"]
        steps += 1

    if reward_vector is None:
        reward_vector = np.zeros((4,), dtype=np.float32)
    return trajectory, reward_vector, steps


def run_parallel_episodes(policy, device, n_envs, max_turns):
    vec_env = vectorizedBig2Games(n_envs)
    pgos, states, avails = vec_env.getCurrStates()
    obs_list = [
        _format_state(pgos[i], states[i], avails[i]) for i in range(n_envs)
    ]
    trajectories = [[] for _ in range(n_envs)]
    reward_vectors = [None for _ in range(n_envs)]
    done = [False for _ in range(n_envs)]

    while not all(done):
        actions = []
        for i in range(n_envs):
            if done[i]:
                actions.append(_random_action(obs_list[i]["action_mask"]))
                continue
            obs = obs_list[i]
            action, logprob, value = select_action(policy, obs, device)
            trajectories[i].append(
                {
                    "obs": obs["obs"].copy(),
                    "mask": obs["action_mask"].copy(),
                    "action": action,
                    "logprob": logprob.detach().cpu().item(),
                    "value": value.detach().cpu().item(),
                    "player": obs["player"],
                }
            )
            actions.append(action)

        rewards, dones, _infos = vec_env.step(actions)
        pgos, states, avails = vec_env.getCurrStates()
        obs_list = [
            _format_state(pgos[i], states[i], avails[i]) for i in range(n_envs)
        ]

        for i in range(n_envs):
            if done[i]:
                continue
            if dones[i]:
                reward_vectors[i] = rewards[i]
                done[i] = True
            if len(trajectories[i]) >= max_turns:
                done[i] = True
                if reward_vectors[i] is None:
                    reward_vectors[i] = np.zeros((4,), dtype=np.float32)

    vec_env.close()
    for i in range(n_envs):
        if reward_vectors[i] is None:
            reward_vectors[i] = np.zeros((4,), dtype=np.float32)
    return trajectories, reward_vectors


def train(args):
    device = torch.device("cpu")
    policy = ActorCriticNet().to(device)

    if args.load and os.path.exists(args.load):
        payload = torch.load(args.load, map_location=device)
        policy.load_state_dict(payload["model_state"])

    optimizer = optim.Adam(policy.parameters(), lr=args.lr)
    os.makedirs(args.save_dir, exist_ok=True)
    log_path = os.path.join(args.save_dir, "ppo_log.csv")
    if not os.path.exists(log_path):
        with open(log_path, "w", encoding="ascii") as f:
            f.write("episode,steps,loss,policy_loss,value_loss,entropy,avg_reward\n")

    start = time.time()
    env = Big2Env() if args.n_envs == 1 else None

    for episode in range(1, args.episodes + 1):
        if args.n_envs == 1:
            trajs, reward_vectors = [], []
            traj, reward_vector, steps = run_episode(
                env, policy, device, args.max_turns
            )
            trajs.append(traj)
            reward_vectors.append(reward_vector)
        else:
            trajs, reward_vectors = run_parallel_episodes(
                policy, device, args.n_envs, args.max_turns
            )
            steps = sum(len(t) for t in trajs)

        all_obs = []
        all_masks = []
        all_actions = []
        all_old_logprobs = []
        all_advs = []
        all_returns = []

        for traj, reward_vector in zip(trajs, reward_vectors):
            rewards = [float(reward_vector[t["player"] - 1]) for t in traj]
            values = [t["value"] for t in traj]
            advantages, returns = compute_gae(rewards, values, args.gamma, args.lam)

            all_obs.extend([t["obs"] for t in traj])
            all_masks.extend([t["mask"] for t in traj])
            all_actions.extend([t["action"] for t in traj])
            all_old_logprobs.extend([t["logprob"] for t in traj])
            all_advs.extend(advantages)
            all_returns.extend(returns)

        obs_tensor = torch.from_numpy(np.stack(all_obs)).to(device)
        mask_tensor = torch.from_numpy(np.stack(all_masks)).to(device)
        action_tensor = torch.tensor(all_actions, device=device)
        old_logprob = torch.tensor(all_old_logprobs, device=device)
        adv_tensor = torch.tensor(all_advs, device=device)
        ret_tensor = torch.tensor(all_returns, device=device)

        adv_tensor = (adv_tensor - adv_tensor.mean()) / (adv_tensor.std() + 1e-8)

        for _ in range(args.epochs):
            logits, value = policy(obs_tensor, mask_tensor)
            dist = torch.distributions.Categorical(logits=logits)
            logprob = dist.log_prob(action_tensor)
            ratio = torch.exp(logprob - old_logprob)
            surr1 = ratio * adv_tensor
            surr2 = torch.clamp(ratio, 1.0 - args.clip, 1.0 + args.clip) * adv_tensor
            policy_loss = -torch.mean(torch.min(surr1, surr2))
            value_loss = torch.mean((ret_tensor - value) ** 2)
            entropy = torch.mean(dist.entropy())
            loss = policy_loss + args.vf_coef * value_loss - args.entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_reward = float(np.mean([np.mean(rv) for rv in reward_vectors]))
        if episode % args.log_every == 0:
            print(
                f"ep={episode} steps={steps} loss={loss.item():.4f} "
                f"policy={policy_loss.item():.4f} value={value_loss.item():.4f} "
                f"entropy={entropy.item():.4f} avg_reward={avg_reward:.2f}"
            )

        with open(log_path, "a", encoding="ascii") as f:
            f.write(
                f"{episode},{steps},{loss.item():.6f},{policy_loss.item():.6f},"
                f"{value_loss.item():.6f},{entropy.item():.6f},{avg_reward:.6f}\n"
            )

        if episode % args.save_every == 0:
            save_path = os.path.join(args.save_dir, f"ppo_ep{episode}.pt")
            torch.save({"model_state": policy.state_dict()}, save_path)

    elapsed = time.time() - start
    print(f"done in {elapsed:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--max-turns", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--save-dir", type=str, default="ml/models")
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--load", type=str, default="")
    train(parser.parse_args())
