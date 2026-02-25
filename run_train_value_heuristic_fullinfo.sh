#!/usr/bin/env bash
set -euo pipefail

EPISODES=${EPISODES:-1000}
BATCH=${BATCH:-64}
UPDATES=${UPDATES:-2}
VAL_SIZE=${VAL_SIZE:-512}
LOG_EVERY=${LOG_EVERY:-10}
DEVICE=${DEVICE:-cpu}
LR=${LR:-1e-3}
VALUE_SCALE=${VALUE_SCALE:-15}
BUFFER_CAPACITY=${BUFFER_CAPACITY:-100000}
ENDGAME_STEPS_THRESHOLD=${ENDGAME_STEPS_THRESHOLD:-8}
ENDGAME_TOTAL_THRESHOLD=${ENDGAME_TOTAL_THRESHOLD:--1}
ENDGAME_MAX_HAND_THRESHOLD=${ENDGAME_MAX_HAND_THRESHOLD:--1}
ENDGAME_OVERSAMPLE=${ENDGAME_OVERSAMPLE:-1}
STEPS_WEIGHT_LE8=${STEPS_WEIGHT_LE8:-4}
STEPS_WEIGHT_LE16=${STEPS_WEIGHT_LE16:-3}
STEPS_WEIGHT_LE24=${STEPS_WEIGHT_LE24:-2}
STEPS_WEIGHT_GT24=${STEPS_WEIGHT_GT24:-1}
SAVE=${SAVE:-ML_SKHU/models/value/value_heuristic_fullinfo.pt}
INIT_CKPT=${INIT_CKPT:-}
SAVE_BEST=${SAVE_BEST:-0}
SAVE_BEST_VAL_MAE=${SAVE_BEST_VAL_MAE:-}
SAVE_BEST_ENDGAME_MAE=${SAVE_BEST_ENDGAME_MAE:-}

ARGS=(
  --episodes "$EPISODES"
  --batch-size "$BATCH"
  --updates-per-episode "$UPDATES"
  --val-size "$VAL_SIZE"
  --log-every "$LOG_EVERY"
  --device "$DEVICE"
  --lr "$LR"
  --value-scale "$VALUE_SCALE"
  --buffer-capacity "$BUFFER_CAPACITY"
  --endgame-steps-threshold "$ENDGAME_STEPS_THRESHOLD"
  --endgame-total-threshold "$ENDGAME_TOTAL_THRESHOLD"
  --endgame-max-hand-threshold "$ENDGAME_MAX_HAND_THRESHOLD"
  --endgame-oversample "$ENDGAME_OVERSAMPLE"
  --steps-weight-le8 "$STEPS_WEIGHT_LE8"
  --steps-weight-le16 "$STEPS_WEIGHT_LE16"
  --steps-weight-le24 "$STEPS_WEIGHT_LE24"
  --steps-weight-gt24 "$STEPS_WEIGHT_GT24"
  --save "$SAVE"
)

if [[ -n "$INIT_CKPT" ]]; then
  ARGS+=(--init-ckpt "$INIT_CKPT")
fi

if [[ "$SAVE_BEST" == "1" ]]; then
  ARGS+=(--save-best)
fi
if [[ -n "$SAVE_BEST_VAL_MAE" ]]; then
  ARGS+=(--save-best-val-mae "$SAVE_BEST_VAL_MAE")
fi
if [[ -n "$SAVE_BEST_ENDGAME_MAE" ]]; then
  ARGS+=(--save-best-endgame-mae "$SAVE_BEST_ENDGAME_MAE")
fi

python -m ML_SKHU.train_value_heuristic_fullinfo "${ARGS[@]}"
