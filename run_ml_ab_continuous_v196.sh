#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")" || exit 1

mkdir -p ML_AB/runs

# This machine has 10 logical cores. Three training threads is roughly 30% of
# total CPU capacity while still allowing useful progress.
CPU_THREADS=${CPU_THREADS:-3}
SLEEP_BETWEEN_CYCLES=${SLEEP_BETWEEN_CYCLES:-30}
LOG_FILE=${LOG_FILE:-ML_AB/runs/continuous_v196.log}
PID_FILE=${PID_FILE:-ML_AB/runs/continuous_v196.pid}
# Rotate recipes across continuous cycles. Passing CYCLES=1 to auto_upgrade is
# intentional for bounded runs, so rotation has to happen in this wrapper.
RECIPES=${RECIPES:-300:24:0.00:3e-5,400:24:0.01:3e-5,300:32:0.00:2e-5}
IFS=',' read -r -a RECIPE_LIST <<< "$RECIPES"
if [[ ${#RECIPE_LIST[@]} -eq 0 ]]; then
  echo "no auto-upgrade recipes configured" >&2
  exit 1
fi

echo "$$" > "$PID_FILE"

export CPU_THREADS
export OMP_NUM_THREADS="$CPU_THREADS"
export MKL_NUM_THREADS="$CPU_THREADS"
export VECLIB_MAXIMUM_THREADS="$CPU_THREADS"
export NUMEXPR_NUM_THREADS="$CPU_THREADS"

echo "continuous v196 training started pid=$$ cpu_threads=$CPU_THREADS" | tee -a "$LOG_FILE"

cycle=0
while true; do
  cycle=$((cycle + 1))
  ts=$(date '+%Y-%m-%d %H:%M:%S %z')
  recipe_index=$(( (cycle - 1) % ${#RECIPE_LIST[@]} ))
  recipe="${RECIPE_LIST[$recipe_index]}"
  echo "=== continuous cycle $cycle start $ts ===" | tee -a "$LOG_FILE"
  echo "recipe[$recipe_index]=$recipe" | tee -a "$LOG_FILE"

  # Keep each candidate cycle bounded. The auto-upgrade gate promotes only when
  # p1_avg_reward improves enough; win rate is never a gate metric.
  RECIPES="$recipe" \
  CYCLES=${CYCLES:-1} \
  EVAL_GAMES=${EVAL_GAMES:-1000} \
  EVAL_SEEDS=${EVAL_SEEDS:-701,702} \
  MIN_DELTA=${MIN_DELTA:-0.05} \
  TRAIN_DEVICE=${TRAIN_DEVICE:-cpu} \
  DEVICE=${DEVICE:-cpu} \
  CPU_THREADS="$CPU_THREADS" \
  ./run_ml_ab_auto_upgrade.sh 2>&1 | tee -a "$LOG_FILE" || true

  echo "=== continuous cycle $cycle end $(date '+%Y-%m-%d %H:%M:%S %z') ===" | tee -a "$LOG_FILE"
  sleep "$SLEEP_BETWEEN_CYCLES"
done
