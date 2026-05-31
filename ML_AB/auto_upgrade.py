import argparse
import json
import os
import shutil
import subprocess
import sys
import time


def run_cmd(cmd):
    print("RUN " + " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.stdout:
        print(proc.stdout, end="", flush=True)
    if proc.returncode != 0:
        if proc.stderr:
            print(proc.stderr, end="", flush=True)
        raise RuntimeError(f"command failed: {' '.join(cmd)}")
    return proc.stdout


def eval_ckpt(ckpt, games, opponent, device, seed):
    out = run_cmd(
        [
            sys.executable,
            "-m",
            "ML_AB.eval",
            "--ckpt",
            ckpt,
            "--games",
            str(games),
            "--opponent",
            opponent,
            "--device",
            device,
            "--seed",
            str(seed),
        ]
    )
    return json.loads(out.strip().splitlines()[-1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--best", default="ML_AB/models/big2_transformer_best.pt")
    parser.add_argument("--current", default="ML_AB/models/big2_transformer_current.pt")
    parser.add_argument("--work-dir", default="ML_AB/models/auto")
    parser.add_argument("--log", default="ML_AB/runs/auto_upgrade.jsonl")
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--simulations", type=int, default=16)
    parser.add_argument("--move-time-limit-sec", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--updates", type=int, default=2)
    parser.add_argument("--eval-games", type=int, default=1000)
    parser.add_argument("--eval-seeds", default="101,102")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--train-device", default="cpu")
    parser.add_argument("--opponent", default="heuristic", choices=["heuristic", "random", "self"])
    parser.add_argument("--min-delta", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument(
        "--recipes",
        default="300:24:0.00:3e-5,400:24:0.01:3e-5,300:32:0.00:2e-5",
        help="comma-separated episodes:simulations:bootstrap_frac:lr recipes. This project now uses the 196-history model line only.",
    )
    args = parser.parse_args()

    os.makedirs(args.work_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.log), exist_ok=True)
    seeds = [int(x) for x in args.eval_seeds.split(",") if x.strip()]
    recipes = []
    for item in args.recipes.split(","):
        ep_s, sim_s, boot_s, lr_s = item.split(":")
        recipes.append(
            {
                "episodes": int(ep_s),
                "simulations": int(sim_s),
                "bootstrap_frac": float(boot_s),
                "lr": float(lr_s),
            }
        )

    for cycle in range(1, args.cycles + 1):
        recipe = recipes[(cycle - 1) % len(recipes)]
        ts = time.strftime("%Y%m%d_%H%M%S")
        candidate = os.path.join(args.work_dir, f"candidate_cycle{cycle:02d}_{ts}.pt")
        metrics = os.path.join(args.work_dir, f"candidate_cycle{cycle:02d}_{ts}.jsonl")
        print(f"=== AUTO CYCLE {cycle}/{args.cycles} ===", flush=True)
        print(f"best={args.best}", flush=True)
        print(f"candidate={candidate}", flush=True)
        print(f"recipe={recipe}", flush=True)

        run_cmd(
            [
                sys.executable,
                "-m",
                "ML_AB.train_search",
                "--init",
                args.best,
                "--save",
                candidate,
                "--metrics",
                metrics,
                "--episodes",
                str(recipe["episodes"]),
                "--simulations",
                str(recipe["simulations"]),
                "--move-time-limit-sec",
                str(args.move_time_limit_sec),
                "--batch-size",
                str(args.batch_size),
                "--updates-per-episode",
                str(args.updates),
                "--bootstrap-frac",
                str(recipe["bootstrap_frac"]),
                "--device",
                args.train_device,
                "--seed",
                str(args.seed + cycle),
                "--lr",
                str(recipe["lr"]),
            ]
        )

        candidate_scores = []
        best_scores = []
        eval_rows = []
        for seed in seeds:
            cand_eval = eval_ckpt(candidate, args.eval_games, args.opponent, args.device, seed)
            best_eval = eval_ckpt(args.best, args.eval_games, args.opponent, args.device, seed)
            candidate_scores.append(float(cand_eval["p1_avg_reward"]))
            best_scores.append(float(best_eval["p1_avg_reward"]))
            eval_rows.append({"seed": seed, "candidate": cand_eval, "best": best_eval})

        cand_mean = sum(candidate_scores) / len(candidate_scores)
        best_mean = sum(best_scores) / len(best_scores)
        delta = cand_mean - best_mean
        accepted = delta >= args.min_delta
        if accepted:
            shutil.copy(candidate, args.best)
            shutil.copy(candidate, args.current)
            print(f"PROMOTED candidate: mean_delta={delta:.4f}", flush=True)
        else:
            print(f"REJECTED candidate: mean_delta={delta:.4f}", flush=True)

        row = {
            "cycle": cycle,
            "candidate": candidate,
            "metrics": metrics,
            "opponent": args.opponent,
            "eval_games": args.eval_games,
            "eval_seeds": seeds,
            "candidate_mean": cand_mean,
            "best_mean": best_mean,
            "delta": delta,
            "min_delta": args.min_delta,
            "accepted": accepted,
            "evals": eval_rows,
            "timestamp": ts,
            "recipe": recipe,
        }
        with open(args.log, "a") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
        print(json.dumps(row, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
