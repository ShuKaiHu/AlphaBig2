#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")" || exit 1
source "$(pwd)/scripts/ensure_repo_venv.sh"

EPISODES=${EPISODES:-2000}
BATCH=${BATCH:-64}
UPDATES=${UPDATES:-2}
MAX_TURNS=${MAX_TURNS:-300}
DEVICE=${DEVICE:-cpu}
SEED=${SEED:-0}
DMODEL=${DMODEL:-128}
LAYERS=${LAYERS:-3}
HEADS=${HEADS:-4}
SAVE=${SAVE:-ML_AB/models/big2_transformer_current.pt}
METRICS=${METRICS:-ML_AB/runs/train_metrics.jsonl}
CPU_THREADS=${CPU_THREADS:-3}

export CPU_THREADS
export OMP_NUM_THREADS="$CPU_THREADS"
export MKL_NUM_THREADS="$CPU_THREADS"
export VECLIB_MAXIMUM_THREADS="$CPU_THREADS"
export NUMEXPR_NUM_THREADS="$CPU_THREADS"

python -m ML_AB.train \
  --episodes "$EPISODES" \
  --batch-size "$BATCH" \
  --updates-per-episode "$UPDATES" \
  --max-turns "$MAX_TURNS" \
  --device "$DEVICE" \
  --seed "$SEED" \
  --d-model "$DMODEL" \
  --layers "$LAYERS" \
  --heads "$HEADS" \
  --save "$SAVE" \
  --metrics "$METRICS"
