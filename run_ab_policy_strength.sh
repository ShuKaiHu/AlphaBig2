#!/usr/bin/env bash
set -euo pipefail

# A/B test policy training strength by sweeping EPISODES / UPDATES.
# Uses a fixed value checkpoint and resets policy.pt to the same baseline before each run.

SAVE_DIR=${SAVE_DIR:-ML_SKHU/models/new_three_cycle}
DEVICE=${DEVICE:-cpu}
SIMS=${SIMS:-120}
EVAL_GAMES=${EVAL_GAMES:-500}
LOG_EVERY=${LOG_EVERY:-20}
FIXED_VALUE_CKPT=${FIXED_VALUE_CKPT:-ML_SKHU/models/value/value_heuristic_fullinfo_tanh_step_ladder.best_endgame_mae.pt}

POLICY_CKPT="$SAVE_DIR/policy.pt"
BELIEF_CKPT="$SAVE_DIR/belief.pt"

for f in "$POLICY_CKPT" "$BELIEF_CKPT" "$FIXED_VALUE_CKPT"; do
  if [[ ! -f "$f" ]]; then
    echo "missing checkpoint: $f"
    exit 1
  fi
done

BASELINE_POLICY_SNAPSHOT="$SAVE_DIR/policy.pt.ab_strength_baseline_snapshot"
cp "$POLICY_CKPT" "$BASELINE_POLICY_SNAPSHOT"

run_one() {
  local tag="$1"
  local episodes="$2"
  local updates="$3"
  local log_file="$4"

  echo "=================================================="
  echo "STRENGTH RUN: $tag"
  echo "EPISODES=$episodes UPDATES=$updates SIMS=$SIMS EVAL_GAMES=$EVAL_GAMES LOG_EVERY=$LOG_EVERY"
  echo "VALUE=$FIXED_VALUE_CKPT"
  echo "LOG=$log_file"
  echo "=================================================="

  cp "$BASELINE_POLICY_SNAPSHOT" "$POLICY_CKPT"

  FIXED_VALUE_CKPT="$FIXED_VALUE_CKPT" \
  SAVE_DIR="$SAVE_DIR" \
  DEVICE="$DEVICE" \
  EPISODES="$episodes" \
  SIMS="$SIMS" \
  UPDATES="$updates" \
  EVAL_GAMES="$EVAL_GAMES" \
  LOG_EVERY="$LOG_EVERY" \
  ./run_train_policy_only_fixed_value.sh | tee "$log_file"
}

# Default sweep (you can edit/add combinations if needed)
run_one "ep100_upd2" 100 2 "ab_policy_strength_ep100_upd2.log"
run_one "ep200_upd2" 200 2 "ab_policy_strength_ep200_upd2.log"
run_one "ep200_upd1" 200 1 "ab_policy_strength_ep200_upd1.log"
run_one "ep400_upd2" 400 2 "ab_policy_strength_ep400_upd2.log"

echo
echo "Done. Compare logs:"
echo "  ab_policy_strength_ep100_upd2.log"
echo "  ab_policy_strength_ep200_upd2.log"
echo "  ab_policy_strength_ep200_upd1.log"
echo "  ab_policy_strength_ep400_upd2.log"

