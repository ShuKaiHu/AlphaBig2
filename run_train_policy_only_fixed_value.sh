#!/usr/bin/env bash
set -euo pipefail

# Train POLICY only with fixed BELIEF (frozen) + fixed VALUE (frozen)
# Uses oracle/full-info belief input for phase2-style training.
# Flow: eval before -> train(policy only) -> eval after

EPISODES=${EPISODES:-400}
SIMS=${SIMS:-120}
BATCH=${BATCH:-64}
UPDATES=${UPDATES:-3}
EVAL_GAMES=${EVAL_GAMES:-500}
LOG_EVERY=${LOG_EVERY:-5}
DEVICE=${DEVICE:-cpu}
OPP=${OPP:-heuristic}
VALUE_SCALE=${VALUE_SCALE:-15}
VAL_SIZE=${VAL_SIZE:-256}
SAVE_DIR=${SAVE_DIR:-ML_SKHU/models/new_three_cycle}

# MCTS target schedule (same as run_train_policy_value.sh defaults)
SELFPLAY_TEMP_START=${SELFPLAY_TEMP_START:-1.0}
SELFPLAY_TEMP_END=${SELFPLAY_TEMP_END:-0.4}
TARGET_TEMP_START=${TARGET_TEMP_START:-1.0}
TARGET_TEMP_END=${TARGET_TEMP_END:-0.35}
DIRICHLET_ALPHA=${DIRICHLET_ALPHA:-0.3}
DIRICHLET_FRAC_START=${DIRICHLET_FRAC_START:-0.25}
DIRICHLET_FRAC_END=${DIRICHLET_FRAC_END:-0.02}

# value model to pin/freeze during policy training
FIXED_VALUE_CKPT=${FIXED_VALUE_CKPT:-ML_SKHU/models/value/value_heuristic_fullinfo.pt}

mkdir -p "$SAVE_DIR"

BELIEF_CKPT="$SAVE_DIR/belief.pt"
POLICY_CKPT="$SAVE_DIR/policy.pt"
VALUE_CKPT="$SAVE_DIR/value.pt"

for f in "$BELIEF_CKPT" "$POLICY_CKPT" "$FIXED_VALUE_CKPT"; do
  if [[ ! -f "$f" ]]; then
    echo "missing checkpoint: $f"
    exit 1
  fi
done

# Keep a backup of current value and then pin the fixed value into pipeline path.
cp "$VALUE_CKPT" "$SAVE_DIR/value.pt.bak_before_policy_only_fixed_value" 2>/dev/null || true
cp "$FIXED_VALUE_CKPT" "$VALUE_CKPT"

echo "=== Eval BEFORE training (fixed value) ==="
python -m ML_SKHU.eval_three_models \
  --belief-ckpt "$BELIEF_CKPT" \
  --policy-ckpt "$POLICY_CKPT" \
  --value-ckpt "$VALUE_CKPT" \
  --games "$EVAL_GAMES" \
  --simulations "$SIMS" \
  --opponent "$OPP" \
  --belief-mode oracle \
  --value-scale "$VALUE_SCALE" \
  --device "$DEVICE"

echo "=== Train POLICY only (belief frozen, value frozen, oracle input) ==="
python -m ML_SKHU.train_three_models \
  --episodes "$EPISODES" \
  --simulations "$SIMS" \
  --batch-size "$BATCH" \
  --updates-per-episode "$UPDATES" \
  --log-every "$LOG_EVERY" \
  --opponent "$OPP" \
  --device "$DEVICE" \
  --value-scale "$VALUE_SCALE" \
  --val-size "$VAL_SIZE" \
  --belief-mode oracle \
  --selfplay-temp-start "$SELFPLAY_TEMP_START" \
  --selfplay-temp-end "$SELFPLAY_TEMP_END" \
  --target-temp-start "$TARGET_TEMP_START" \
  --target-temp-end "$TARGET_TEMP_END" \
  --dirichlet-alpha "$DIRICHLET_ALPHA" \
  --dirichlet-frac-start "$DIRICHLET_FRAC_START" \
  --dirichlet-frac-end "$DIRICHLET_FRAC_END" \
  --belief-ckpt "$BELIEF_CKPT" \
  --policy-ckpt "$POLICY_CKPT" \
  --value-ckpt "$VALUE_CKPT" \
  --policy-weight 1.0 \
  --value-weight 0.0 \
  --save-dir "$SAVE_DIR"

# Re-pin fixed value in case train_three_models re-saved a changed value checkpoint.
cp "$FIXED_VALUE_CKPT" "$VALUE_CKPT"

echo "=== Eval AFTER training (fixed value) ==="
python -m ML_SKHU.eval_three_models \
  --belief-ckpt "$BELIEF_CKPT" \
  --policy-ckpt "$POLICY_CKPT" \
  --value-ckpt "$VALUE_CKPT" \
  --games "$EVAL_GAMES" \
  --simulations "$SIMS" \
  --opponent "$OPP" \
  --belief-mode oracle \
  --value-scale "$VALUE_SCALE" \
  --device "$DEVICE"
