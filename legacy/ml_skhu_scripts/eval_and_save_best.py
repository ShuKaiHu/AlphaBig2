#!/usr/bin/env python3
"""
Evaluate model and save if it's the strongest based on avg_reward
"""
import argparse
import json
import os
import shutil
import subprocess
import sys


def run_eval(belief_ckpt, policy_ckpt, opponent, games=100, simulations=25):
    """Run evaluation and return avg_reward for all 4 players"""
    cmd = [
        sys.executable,
        "-m",
        "ML_SKHU.eval",
        "--games",
        str(games),
        "--simulations",
        str(simulations),
        "--belief-ckpt",
        belief_ckpt,
        "--policy-ckpt",
        policy_ckpt,
        "--opponent",
        opponent,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Evaluation failed:\n{result.stderr}")
        return None
    
    # Parse output to find avg_reward lines
    output = result.stdout
    print(output)
    
    avg_rewards = {}
    for line in output.split('\n'):
        if 'avg_reward:' in line:
            # Parse: "player1 win_rate: X avg_reward: Y.YY"
            parts = line.split()
            if len(parts) >= 4:
                player = parts[0]
                reward = float(parts[-1])
                avg_rewards[player] = reward
    
    if not avg_rewards:
        print("Could not parse avg_reward from evaluation output")
        return None
    
    # Player 1 (MCTS agent) is the main focus
    return avg_rewards.get("player1", None)


def get_best_reward_file(opponent):
    """Get path to best reward tracking file"""
    return f"ML_SKHU/models/.best_{opponent}_reward.json"


def load_best_reward(opponent):
    """Load best reward for opponent from tracking file"""
    best_file = get_best_reward_file(opponent)
    if os.path.exists(best_file):
        try:
            with open(best_file, 'r') as f:
                data = json.load(f)
                return data.get("best_reward", float('-inf'))
        except Exception as e:
            print(f"Warning: could not load best reward: {e}")
    return float('-inf')


def save_best_reward(opponent, reward):
    """Save best reward to tracking file"""
    best_file = get_best_reward_file(opponent)
    os.makedirs(os.path.dirname(best_file), exist_ok=True)
    with open(best_file, 'w') as f:
        json.dump({"best_reward": reward}, f)


def save_best_model(opponent, belief_src, policy_src):
    """Copy model to best/ directory"""
    best_dir = f"ML_SKHU/models/best_{opponent}"
    os.makedirs(best_dir, exist_ok=True)
    
    belief_dst = f"{best_dir}/belief.pt"
    policy_dst = f"{best_dir}/policy_value.pt"
    
    shutil.copy(belief_src, belief_dst)
    shutil.copy(policy_src, policy_dst)
    
    print(f"✓ Best model saved to {best_dir}/")
    return belief_dst, policy_dst


def main():
    parser = argparse.ArgumentParser(description="Evaluate and save best models")
    parser.add_argument("--opponent", type=str, required=True, choices=["random", "heuristic"])
    parser.add_argument("--belief-ckpt", type=str, required=True)
    parser.add_argument("--policy-ckpt", type=str, required=True)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--simulations", type=int, default=25)
    args = parser.parse_args()
    
    print(f"\n{'='*50}")
    print(f"Evaluating {args.opponent} opponent model...")
    print(f"{'='*50}")
    
    # Run evaluation
    player1_reward = run_eval(
        args.belief_ckpt,
        args.policy_ckpt,
        args.opponent,
        games=args.games,
        simulations=args.simulations,
    )
    
    if player1_reward is None:
        print("Evaluation failed, skipping best model check")
        return
    
    print(f"\nPlayer1 (MCTS) avg_reward: {player1_reward:.2f}")
    
    # Check if this is the best
    best_reward = load_best_reward(args.opponent)
    print(f"Previous best reward: {best_reward:.2f}")
    
    if player1_reward > best_reward:
        print(f"✓ NEW BEST MODEL! (+{player1_reward - best_reward:.2f})")
        save_best_reward(args.opponent, player1_reward)
        save_best_model(args.opponent, args.belief_ckpt, args.policy_ckpt)
    else:
        print(f"Not better than best (difference: {player1_reward - best_reward:.2f})")
    
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
