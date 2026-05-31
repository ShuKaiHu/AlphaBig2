#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")" || exit 1
source "$(pwd)/scripts/ensure_repo_venv.sh"

INIT=${INIT:-ML_AB/models/big2_transformer_current.pt}
SAVE=${SAVE:-ML_AB/models/big2_transformer_search.pt}
METRICS=${METRICS:-ML_AB/runs/search_metrics.jsonl}
EPISODES=${EPISODES:-200}
SIMS=${SIMS:-24}
BATCH=${BATCH:-32}
UPDATES=${UPDATES:-2}
DEVICE=${DEVICE:-mps}
SEED=${SEED:-123}
MCTS_MOVE_TIME_LIMIT=${MCTS_MOVE_TIME_LIMIT:-3.0}
CPU_THREADS=${CPU_THREADS:-3}

export CPU_THREADS
export OMP_NUM_THREADS="$CPU_THREADS"
export MKL_NUM_THREADS="$CPU_THREADS"
export VECLIB_MAXIMUM_THREADS="$CPU_THREADS"
export NUMEXPR_NUM_THREADS="$CPU_THREADS"

python -m ML_AB.train_search \
  --init "$INIT" \
  --save "$SAVE" \
  --metrics "$METRICS" \
  --episodes "$EPISODES" \
  --simulations "$SIMS" \
  --move-time-limit-sec "$MCTS_MOVE_TIME_LIMIT" \
  --batch-size "$BATCH" \
  --updates-per-episode "$UPDATES" \
  --device "$DEVICE" \
  --seed "$SEED"
