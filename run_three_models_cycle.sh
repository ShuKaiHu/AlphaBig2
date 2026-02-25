#!/usr/bin/env bash
set -euo pipefail

# Simple schedule:
# 1) Belief-only warmup
# 2) Policy+Value training (belief frozen)
# 3) Final eval

EP_BELIEF_1=${EP_BELIEF_1:-20}
EP_PV_1=${EP_PV_1:-40}
SIMS=${SIMS:-80}
BATCH=${BATCH:-64}
UPDATES=${UPDATES:-2}
OPP=${OPP:-heuristic}
DEVICE=${DEVICE:-cpu}
SAVE_DIR=${SAVE_DIR:-ML_SKHU/models/new_three_cycle}
EVAL_GAMES=${EVAL_GAMES:-50}
LOG_EVERY=${LOG_EVERY:-5}
VALUE_SCALE=${VALUE_SCALE:-15}
VAL_SIZE=${VAL_SIZE:-256}

mkdir -p "$SAVE_DIR"

echo "=== Phase 1: belief-only warmup ==="
python -m ML_SKHU.train_three_models \
  --episodes "$EP_BELIEF_1" \
  --simulations "$SIMS" \
  --batch-size "$BATCH" \
  --updates-per-episode "$UPDATES" \
  --log-every "$LOG_EVERY" \
  --opponent "$OPP" \
  --device "$DEVICE" \
  --train-belief \
  --value-scale "$VALUE_SCALE" \
  --val-size "$VAL_SIZE" \
  --belief-mode model \
  --policy-weight 0.0 \
  --value-weight 0.0 \
  --save-dir "$SAVE_DIR"

echo "=== Phase 2: policy+value (belief frozen) ==="
python -m ML_SKHU.train_three_models \
  --episodes "$EP_PV_1" \
  --simulations "$SIMS" \
  --batch-size "$BATCH" \
  --updates-per-episode "$UPDATES" \
  --log-every "$LOG_EVERY" \
  --opponent "$OPP" \
  --device "$DEVICE" \
  --value-scale "$VALUE_SCALE" \
  --val-size "$VAL_SIZE" \
  --belief-mode oracle \
  --belief-ckpt "$SAVE_DIR/belief.pt" \
  --policy-ckpt "$SAVE_DIR/policy.pt" \
  --value-ckpt "$SAVE_DIR/value.pt" \
  --save-dir "$SAVE_DIR"

echo "=== Final eval ==="
python -m ML_SKHU.eval_three_models \
  --belief-ckpt "$SAVE_DIR/belief.pt" \
  --policy-ckpt "$SAVE_DIR/policy.pt" \
  --value-ckpt "$SAVE_DIR/value.pt" \
  --games "$EVAL_GAMES" \
  --simulations "$SIMS" \
  --opponent "$OPP" \
  --value-scale "$VALUE_SCALE" \
  --device "$DEVICE"
