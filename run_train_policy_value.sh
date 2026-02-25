#!/usr/bin/env bash
set -euo pipefail

# Train policy+value only (belief frozen, oracle/full-info input):
# 1) eval before
# 2) train
# 3) eval after

EP_PV=${EP_PV:-40}
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
SELFPLAY_TEMP_START=${SELFPLAY_TEMP_START:-1.0}
SELFPLAY_TEMP_END=${SELFPLAY_TEMP_END:-0.4}
TARGET_TEMP_START=${TARGET_TEMP_START:-1.0}
TARGET_TEMP_END=${TARGET_TEMP_END:-0.35}
DIRICHLET_ALPHA=${DIRICHLET_ALPHA:-0.3}
DIRICHLET_FRAC_START=${DIRICHLET_FRAC_START:-0.25}
DIRICHLET_FRAC_END=${DIRICHLET_FRAC_END:-0.02}

mkdir -p "$SAVE_DIR"

BELIEF_CKPT="$SAVE_DIR/belief.pt"
POLICY_CKPT="$SAVE_DIR/policy.pt"
VALUE_CKPT="$SAVE_DIR/value.pt"

if [[ ! -f "$BELIEF_CKPT" ]]; then
  echo "missing checkpoint: $BELIEF_CKPT"
  echo "請先準備 belief.pt（例如先跑 belief warmup）。"
  exit 1
fi
if [[ ! -f "$POLICY_CKPT" ]]; then
  echo "missing checkpoint: $POLICY_CKPT"
  echo "請先準備 policy.pt。"
  exit 1
fi
if [[ ! -f "$VALUE_CKPT" ]]; then
  echo "missing checkpoint: $VALUE_CKPT"
  echo "請先準備 value.pt。"
  exit 1
fi

echo "=== Eval BEFORE training ==="
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

echo "=== Train policy+value (belief frozen, oracle input) ==="
python -m ML_SKHU.train_three_models \
  --episodes "$EP_PV" \
  --simulations "$SIMS" \
  --batch-size "$BATCH" \
  --updates-per-episode "$UPDATES" \
  --log-every "$LOG_EVERY" \
  --opponent "$OPP" \
  --device "$DEVICE" \
  --value-scale "$VALUE_SCALE" \
  --val-size "$VAL_SIZE" \
  --selfplay-temp-start "$SELFPLAY_TEMP_START" \
  --selfplay-temp-end "$SELFPLAY_TEMP_END" \
  --target-temp-start "$TARGET_TEMP_START" \
  --target-temp-end "$TARGET_TEMP_END" \
  --dirichlet-alpha "$DIRICHLET_ALPHA" \
  --dirichlet-frac-start "$DIRICHLET_FRAC_START" \
  --dirichlet-frac-end "$DIRICHLET_FRAC_END" \
  --belief-mode oracle \
  --belief-ckpt "$BELIEF_CKPT" \
  --policy-ckpt "$POLICY_CKPT" \
  --value-ckpt "$VALUE_CKPT" \
  --save-dir "$SAVE_DIR"

echo "=== Eval AFTER training ==="
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
