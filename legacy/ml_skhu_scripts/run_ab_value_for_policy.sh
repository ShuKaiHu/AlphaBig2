#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")" || exit 1
source "$(pwd)/scripts/ensure_repo_venv.sh"

# A/B test multiple value checkpoints as the fixed value teacher for policy-only training.
# It resets policy.pt to the same baseline before each run for fair comparison.

SAVE_DIR=${SAVE_DIR:-ML_SKHU/models/new_three_cycle}
DEVICE=${DEVICE:-cpu}
EPISODES=${EPISODES:-400}
SIMS=${SIMS:-120}
UPDATES=${UPDATES:-3}
EVAL_GAMES=${EVAL_GAMES:-500}
LOG_EVERY=${LOG_EVERY:-5}

POLICY_CKPT="$SAVE_DIR/policy.pt"
BELIEF_CKPT="$SAVE_DIR/belief.pt"

if [[ ! -f "$POLICY_CKPT" ]]; then
  echo "missing checkpoint: $POLICY_CKPT"
  exit 1
fi
if [[ ! -f "$BELIEF_CKPT" ]]; then
  echo "missing checkpoint: $BELIEF_CKPT"
  exit 1
fi

BASELINE_POLICY_SNAPSHOT="$SAVE_DIR/policy.pt.ab_baseline_snapshot"
cp "$POLICY_CKPT" "$BASELINE_POLICY_SNAPSHOT"

VALUE_CURRENT=${VALUE_CURRENT:-ML_SKHU/models/value/value_current.pt}
VALUE_BEST_VAL=${VALUE_BEST_VAL:-ML_SKHU/models/value/value_heuristic_fullinfo_tanh_step_ladder.best_val_mae.pt}
VALUE_BEST_END=${VALUE_BEST_END:-ML_SKHU/models/value/value_heuristic_fullinfo_tanh_step_ladder.best_endgame_mae.pt}

run_one() {
  local tag="$1"
  local value_ckpt="$2"
  local log_file="$3"

  if [[ ! -f "$value_ckpt" ]]; then
    echo "skip $tag (missing value ckpt): $value_ckpt"
    return 0
  fi

  echo "=================================================="
  echo "A/B RUN: $tag"
  echo "VALUE: $value_ckpt"
  echo "LOG:   $log_file"
  echo "=================================================="

  # Reset policy to identical baseline before each test
  cp "$BASELINE_POLICY_SNAPSHOT" "$POLICY_CKPT"

  FIXED_VALUE_CKPT="$value_ckpt" \
  SAVE_DIR="$SAVE_DIR" \
  DEVICE="$DEVICE" \
  EPISODES="$EPISODES" \
  SIMS="$SIMS" \
  UPDATES="$UPDATES" \
  EVAL_GAMES="$EVAL_GAMES" \
  LOG_EVERY="$LOG_EVERY" \
  ./run_train_policy_only_fixed_value.sh | tee "$log_file"
}

run_one "baseline_value_current" "$VALUE_CURRENT" "ab_policy_with_value_current.log"
run_one "candidate_best_val_mae" "$VALUE_BEST_VAL" "ab_policy_with_value_best_val_mae.log"
run_one "candidate_best_endgame_val_mae" "$VALUE_BEST_END" "ab_policy_with_value_best_endgame_mae.log"

echo
echo "Done. Compare these logs:"
echo "  ab_policy_with_value_current.log"
echo "  ab_policy_with_value_best_val_mae.log"
echo "  ab_policy_with_value_best_endgame_mae.log"
