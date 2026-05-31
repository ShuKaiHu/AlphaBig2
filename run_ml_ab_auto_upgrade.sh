#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")" || exit 1
source "$(pwd)/scripts/ensure_repo_venv.sh"

CYCLES=${CYCLES:-3}
EPISODES=${EPISODES:-200}
SIMS=${SIMS:-16}
BATCH=${BATCH:-32}
UPDATES=${UPDATES:-2}
EVAL_GAMES=${EVAL_GAMES:-1000}
EVAL_SEEDS=${EVAL_SEEDS:-101,102}
MIN_DELTA=${MIN_DELTA:-0.10}
DEVICE=${DEVICE:-cpu}
TRAIN_DEVICE=${TRAIN_DEVICE:-cpu}
SEED=${SEED:-1000}
MCTS_MOVE_TIME_LIMIT=${MCTS_MOVE_TIME_LIMIT:-3.0}
# Default recipes are tuned for the v196 line's promotion gate:
# improve p1_avg_reward, not win rate. Keep bootstrap low so candidates do
# not repeatedly drift back toward heuristic/random behavior.
RECIPES=${RECIPES:-300:24:0.00:3e-5,400:24:0.01:3e-5,300:32:0.00:2e-5}
CPU_THREADS=${CPU_THREADS:-3}

export CPU_THREADS
export OMP_NUM_THREADS="$CPU_THREADS"
export MKL_NUM_THREADS="$CPU_THREADS"
export VECLIB_MAXIMUM_THREADS="$CPU_THREADS"
export NUMEXPR_NUM_THREADS="$CPU_THREADS"

python -m ML_AB.auto_upgrade \
  --cycles "$CYCLES" \
  --episodes "$EPISODES" \
  --simulations "$SIMS" \
  --move-time-limit-sec "$MCTS_MOVE_TIME_LIMIT" \
  --batch-size "$BATCH" \
  --updates "$UPDATES" \
  --eval-games "$EVAL_GAMES" \
  --eval-seeds "$EVAL_SEEDS" \
  --min-delta "$MIN_DELTA" \
  --device "$DEVICE" \
  --train-device "$TRAIN_DEVICE" \
  --seed "$SEED" \
  --recipes "$RECIPES"
