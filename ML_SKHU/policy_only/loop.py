import argparse
import os
import subprocess


def run_once(run_dir, prev_policy=None, args=None):
    os.makedirs(run_dir, exist_ok=True)
    data_path = os.path.join(run_dir, "policy_data.npz")
    policy_path = os.path.join(run_dir, "policy_only.pt")
    log_path = os.path.join(run_dir, "eval_log.txt")

    cmd_selfplay = [
        "python",
        "-m",
        "ML_SKHU.policy_only.policy_selfplay",
        "--games",
        str(args.games),
        "--out",
        data_path,
    ]
    subprocess.run(cmd_selfplay, check=True)

    train_cmd = [
        "python",
        "-m",
        "ML_SKHU.policy_only.train_policy_only",
        "--data",
        data_path,
        "--epochs",
        str(args.epochs),
        "--value-weight",
        str(args.value_weight),
        "--save",
        policy_path,
    ]
    if args.reinforce:
        train_cmd.append("--reinforce")
    subprocess.run(train_cmd, check=True)

    eval_cmd = [
        "python",
        "-m",
        "ML_SKHU.policy_only.eval_policy_only",
        "--policy-ckpt",
        policy_path,
        "--games",
        str(args.eval_games),
    ]
    with open(log_path, "w") as fh:
        subprocess.run(eval_cmd, check=True, stdout=fh)

    return policy_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default="policy_only_runs")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--games", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--value-weight", type=float, default=3.0)
    parser.add_argument("--reinforce", action="store_true")
    parser.add_argument("--eval-games", type=int, default=50)
    args = parser.parse_args()

    os.makedirs(args.base_dir, exist_ok=True)
    prev_policy = None
    for run in range(1, args.runs + 1):
        run_dir = os.path.join(args.base_dir, f"run{run}")
        print(f"=== starting run {run} -> {run_dir} ===")
        prev_policy = run_once(run_dir, prev_policy, args)


if __name__ == "__main__":
    main()
