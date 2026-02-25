#!/usr/bin/env bash
set -euo pipefail

# ============================================
# Policy / Value Alternating Iteration Pipeline
# ============================================
#
# Current version:
# - Policy step: belief frozen + value frozen + oracle input
# - Value step : heuristic-only + full-info supervision
# - Each round saves timestamp checkpoints into:
#     ML_SKHU/models/policy/
#     ML_SKHU/models/value/
#
# Notes:
# - Policy step applies an automatic gate based on player1 avg_reward
#   (Eval AFTER must beat Eval BEFORE by threshold to promote policy_current.pt).
# - Timestamp checkpoints are always saved for later review.
#

# -------------------------
# Global iteration settings
# -------------------------
ROUNDS=${ROUNDS:-3}
DEVICE=${DEVICE:-cpu}
SAVE_DIR=${SAVE_DIR:-ML_SKHU/models/new_three_cycle}
VALUE_SCALE=${VALUE_SCALE:-15}

# -------------------------
# Policy step settings
# -------------------------
POLICY_EPISODES=${POLICY_EPISODES:-400}
POLICY_SIMS=${POLICY_SIMS:-120}
POLICY_BATCH=${POLICY_BATCH:-64}
POLICY_UPDATES=${POLICY_UPDATES:-3}
POLICY_EVAL_GAMES=${POLICY_EVAL_GAMES:-500}
POLICY_LOG_EVERY=${POLICY_LOG_EVERY:-100}
POLICY_VAL_SIZE=${POLICY_VAL_SIZE:-256}
POLICY_OPP=${POLICY_OPP:-heuristic}
POLICY_GATE_MIN_DELTA=${POLICY_GATE_MIN_DELTA:-0.0}

# MCTS target schedule for policy training
SELFPLAY_TEMP_START=${SELFPLAY_TEMP_START:-1.0}
SELFPLAY_TEMP_END=${SELFPLAY_TEMP_END:-0.4}
TARGET_TEMP_START=${TARGET_TEMP_START:-1.0}
TARGET_TEMP_END=${TARGET_TEMP_END:-0.35}
DIRICHLET_ALPHA=${DIRICHLET_ALPHA:-0.3}
DIRICHLET_FRAC_START=${DIRICHLET_FRAC_START:-0.25}
DIRICHLET_FRAC_END=${DIRICHLET_FRAC_END:-0.02}

# Fixed value used during policy training
FIXED_VALUE_CKPT=${FIXED_VALUE_CKPT:-ML_SKHU/models/value/value_heuristic_fullinfo.pt}

# -------------------------
# Value step settings
# -------------------------
VALUE_EPISODES=${VALUE_EPISODES:-1000}
VALUE_BATCH=${VALUE_BATCH:-64}
VALUE_UPDATES=${VALUE_UPDATES:-2}
VALUE_VAL_SIZE=${VALUE_VAL_SIZE:-512}
VALUE_LOG_EVERY=${VALUE_LOG_EVERY:-100}
VALUE_LR=${VALUE_LR:-1e-3}
VALUE_BUFFER_CAPACITY=${VALUE_BUFFER_CAPACITY:-100000}
VALUE_GATE_ENABLE=${VALUE_GATE_ENABLE:-1}
VALUE_GATE_MIN_IMPROVE=${VALUE_GATE_MIN_IMPROVE:-0.003}

# If set, value step initializes from previous value checkpoint each round.
VALUE_INIT_FROM_CURRENT=${VALUE_INIT_FROM_CURRENT:-1}

# -------------------------
# Paths / aliases
# -------------------------
BELIEF_DIR=${BELIEF_DIR:-ML_SKHU/models/belief}
POLICY_DIR=${POLICY_DIR:-ML_SKHU/models/policy}
VALUE_DIR=${VALUE_DIR:-ML_SKHU/models/value}
NOTES_FILE=${NOTES_FILE:-ML_SKHU/models/checkpoint_notes.md}
VALUE_META_FILE=${VALUE_META_FILE:-$VALUE_DIR/value_current_meta.txt}

BELIEF_CURRENT=${BELIEF_CURRENT:-$BELIEF_DIR/belief_current.pt}
POLICY_CURRENT=${POLICY_CURRENT:-$POLICY_DIR/policy_current.pt}
VALUE_CURRENT=${VALUE_CURRENT:-$VALUE_DIR/value_current.pt}

mkdir -p "$BELIEF_DIR" "$POLICY_DIR" "$VALUE_DIR" "$SAVE_DIR"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "missing file: $1"
    exit 1
  fi
}

timestamp() {
  date +%Y%m%d_%H%M%S
}

ensure_current_aliases() {
  # belief
  if [[ ! -f "$BELIEF_CURRENT" && -f "$SAVE_DIR/belief.pt" ]]; then
    cp "$SAVE_DIR/belief.pt" "$BELIEF_CURRENT"
  fi
  # policy
  if [[ ! -f "$POLICY_CURRENT" && -f "$SAVE_DIR/policy.pt" ]]; then
    cp "$SAVE_DIR/policy.pt" "$POLICY_CURRENT"
  fi
  # value
  if [[ ! -f "$VALUE_CURRENT" ]]; then
    if [[ -f "$FIXED_VALUE_CKPT" ]]; then
      cp "$FIXED_VALUE_CKPT" "$VALUE_CURRENT"
    elif [[ -f "$SAVE_DIR/value.pt" ]]; then
      cp "$SAVE_DIR/value.pt" "$VALUE_CURRENT"
    fi
  fi
}

ensure_current_aliases

require_file "$BELIEF_CURRENT"
require_file "$POLICY_CURRENT"
require_file "$VALUE_CURRENT"

echo "=== Iteration Pipeline Config ==="
echo "ROUNDS=$ROUNDS DEVICE=$DEVICE SAVE_DIR=$SAVE_DIR VALUE_SCALE=$VALUE_SCALE"
echo "POLICY: episodes=$POLICY_EPISODES sims=$POLICY_SIMS batch=$POLICY_BATCH updates=$POLICY_UPDATES eval_games=$POLICY_EVAL_GAMES log_every=$POLICY_LOG_EVERY"
echo "POLICY GATE: min_delta_avg_reward=$POLICY_GATE_MIN_DELTA (player1)"
echo "POLICY MCTS schedule: selfplay_temp=$SELFPLAY_TEMP_START->$SELFPLAY_TEMP_END target_temp=$TARGET_TEMP_START->$TARGET_TEMP_END noise=$DIRICHLET_FRAC_START->$DIRICHLET_FRAC_END alpha=$DIRICHLET_ALPHA"
echo "VALUE: episodes=$VALUE_EPISODES batch=$VALUE_BATCH updates=$VALUE_UPDATES val_size=$VALUE_VAL_SIZE log_every=$VALUE_LOG_EVERY lr=$VALUE_LR buffer=$VALUE_BUFFER_CAPACITY"
echo "VALUE GATE: enable=$VALUE_GATE_ENABLE min_improve_mae=$VALUE_GATE_MIN_IMPROVE (lower is better)"
echo "CURRENT ALIASES:"
echo "  belief_current=$BELIEF_CURRENT"
echo "  policy_current=$POLICY_CURRENT"
echo "  value_current=$VALUE_CURRENT"
echo

for ROUND in $(seq 1 "$ROUNDS"); do
  echo "========================================"
  echo "ROUND $ROUND / $ROUNDS"
  echo "========================================"

  # Sync working dir checkpoints from current aliases
  cp "$BELIEF_CURRENT" "$SAVE_DIR/belief.pt"
  cp "$POLICY_CURRENT" "$SAVE_DIR/policy.pt"
  cp "$VALUE_CURRENT" "$SAVE_DIR/value.pt"

  echo "=== [Round $ROUND] Policy step (fixed value) ==="
  POLICY_STEP_LOG=$(mktemp -t policy_step_round${ROUND}.XXXXXX.log)
  EPISODES="$POLICY_EPISODES" \
  SIMS="$POLICY_SIMS" \
  BATCH="$POLICY_BATCH" \
  UPDATES="$POLICY_UPDATES" \
  EVAL_GAMES="$POLICY_EVAL_GAMES" \
  LOG_EVERY="$POLICY_LOG_EVERY" \
  DEVICE="$DEVICE" \
  OPP="$POLICY_OPP" \
  VALUE_SCALE="$VALUE_SCALE" \
  VAL_SIZE="$POLICY_VAL_SIZE" \
  SELFPLAY_TEMP_START="$SELFPLAY_TEMP_START" \
  SELFPLAY_TEMP_END="$SELFPLAY_TEMP_END" \
  TARGET_TEMP_START="$TARGET_TEMP_START" \
  TARGET_TEMP_END="$TARGET_TEMP_END" \
  DIRICHLET_ALPHA="$DIRICHLET_ALPHA" \
  DIRICHLET_FRAC_START="$DIRICHLET_FRAC_START" \
  DIRICHLET_FRAC_END="$DIRICHLET_FRAC_END" \
  FIXED_VALUE_CKPT="$VALUE_CURRENT" \
  SAVE_DIR="$SAVE_DIR" \
  ./run_train_policy_only_fixed_value.sh | tee "$POLICY_STEP_LOG"

  TS_P=$(timestamp)
  POLICY_TS_CKPT="$POLICY_DIR/policy_iter$(printf '%02d' "$ROUND")_${TS_P}.pt"
  cp "$SAVE_DIR/policy.pt" "$POLICY_TS_CKPT"
  echo "[Round $ROUND] saved policy checkpoint: $POLICY_TS_CKPT"

  _P1_LINES_STR=$(grep 'player1 win_rate:' "$POLICY_STEP_LOG" || true)
  POLICY_BEFORE_REWARD=""
  POLICY_AFTER_REWARD=""
  _P1_LINE_COUNT=$(printf "%s\n" "$_P1_LINES_STR" | sed '/^$/d' | wc -l | tr -d ' ')
  if [[ "${_P1_LINE_COUNT:-0}" -ge 2 ]]; then
    POLICY_BEFORE_REWARD=$(printf "%s\n" "$_P1_LINES_STR" | sed -n '1p' | awk '{print $NF}')
    POLICY_AFTER_REWARD=$(printf "%s\n" "$_P1_LINES_STR" | sed -n '2p' | awk '{print $NF}')
  fi

  POLICY_GATE_STATUS="unknown"
  POLICY_GATE_DELTA="n/a"
  if [[ -n "$POLICY_BEFORE_REWARD" && -n "$POLICY_AFTER_REWARD" ]]; then
    POLICY_GATE_DELTA=$(python3 - <<PY
b=float("$POLICY_BEFORE_REWARD")
a=float("$POLICY_AFTER_REWARD")
print(f"{a-b:.6f}")
PY
)
    POLICY_GATE_PASS=$(python3 - <<PY
b=float("$POLICY_BEFORE_REWARD")
a=float("$POLICY_AFTER_REWARD")
t=float("$POLICY_GATE_MIN_DELTA")
print("1" if (a >= b + t) else "0")
PY
)
    if [[ "$POLICY_GATE_PASS" == "1" ]]; then
      cp "$SAVE_DIR/policy.pt" "$POLICY_CURRENT"
      POLICY_GATE_STATUS="accepted"
      echo "[Round $ROUND] POLICY GATE: ACCEPTED (before=$POLICY_BEFORE_REWARD after=$POLICY_AFTER_REWARD delta=$POLICY_GATE_DELTA)"
    else
      POLICY_GATE_STATUS="rejected"
      echo "[Round $ROUND] POLICY GATE: REJECTED (before=$POLICY_BEFORE_REWARD after=$POLICY_AFTER_REWARD delta=$POLICY_GATE_DELTA)"
      echo "[Round $ROUND] policy_current remains unchanged: $POLICY_CURRENT"
    fi
  else
    # Fallback: if parsing failed, do not promote automatically.
    POLICY_GATE_STATUS="parse_failed"
    echo "[Round $ROUND] POLICY GATE: PARSE FAILED (could not read before/after player1 avg_reward)."
    echo "[Round $ROUND] policy_current remains unchanged: $POLICY_CURRENT"
  fi

  # Optional: preserve the fixed value snapshot used in this round
  VALUE_FIXED_SNAPSHOT="$VALUE_DIR/value_used_for_policy_iter$(printf '%02d' "$ROUND")_${TS_P}.pt"
  cp "$VALUE_CURRENT" "$VALUE_FIXED_SNAPSHOT"
  echo "[Round $ROUND] snapshot fixed value used in policy step: $VALUE_FIXED_SNAPSHOT"

  echo "=== [Round $ROUND] Value step (heuristic + full-info) ==="
  VALUE_SAVE_TS="$VALUE_DIR/value_iter$(printf '%02d' "$ROUND")_$(timestamp).pt"
  VALUE_INIT_ARG=""
  if [[ "$VALUE_INIT_FROM_CURRENT" == "1" ]]; then
    VALUE_INIT_ARG="$VALUE_CURRENT"
  fi

  VALUE_STEP_LOG=$(mktemp -t value_step_round${ROUND}.XXXXXX.log)
  EPISODES="$VALUE_EPISODES" \
  BATCH="$VALUE_BATCH" \
  UPDATES="$VALUE_UPDATES" \
  VAL_SIZE="$VALUE_VAL_SIZE" \
  LOG_EVERY="$VALUE_LOG_EVERY" \
  DEVICE="$DEVICE" \
  LR="$VALUE_LR" \
  VALUE_SCALE="$VALUE_SCALE" \
  BUFFER_CAPACITY="$VALUE_BUFFER_CAPACITY" \
  SAVE="$VALUE_SAVE_TS" \
  INIT_CKPT="$VALUE_INIT_ARG" \
  ./run_train_value_heuristic_fullinfo.sh | tee "$VALUE_STEP_LOG"

  echo "[Round $ROUND] saved value checkpoint: $VALUE_SAVE_TS"

  VALUE_NEW_MAE=$(grep 'val_mae=' "$VALUE_STEP_LOG" | tail -n 1 | sed -E 's/.*val_mae=([0-9.]+).*/\1/' || true)
  VALUE_OLD_MAE=""
  if [[ -f "$VALUE_META_FILE" ]]; then
    VALUE_OLD_MAE=$(grep '^val_mae=' "$VALUE_META_FILE" | tail -n 1 | cut -d= -f2 || true)
  fi
  VALUE_GATE_STATUS="disabled"
  VALUE_GATE_DELTA="n/a"

  if [[ "$VALUE_GATE_ENABLE" == "1" ]]; then
    if [[ -z "$VALUE_NEW_MAE" ]]; then
      VALUE_GATE_STATUS="parse_failed"
      echo "[Round $ROUND] VALUE GATE: PARSE FAILED (could not find val_mae), value_current remains unchanged."
    elif [[ -z "$VALUE_OLD_MAE" ]]; then
      cp "$VALUE_SAVE_TS" "$VALUE_CURRENT"
      {
        echo "val_mae=$VALUE_NEW_MAE"
        echo "source=$VALUE_SAVE_TS"
        echo "updated_at=$(date '+%Y-%m-%d %H:%M:%S')"
      } > "$VALUE_META_FILE"
      VALUE_GATE_STATUS="accepted_no_baseline"
      echo "[Round $ROUND] VALUE GATE: ACCEPTED (no baseline; new val_mae=$VALUE_NEW_MAE)"
    else
      VALUE_GATE_DELTA=$(python3 - <<PY
old=float("$VALUE_OLD_MAE")
new=float("$VALUE_NEW_MAE")
print(f"{old-new:.6f}")
PY
)
      VALUE_GATE_PASS=$(python3 - <<PY
old=float("$VALUE_OLD_MAE")
new=float("$VALUE_NEW_MAE")
t=float("$VALUE_GATE_MIN_IMPROVE")
print("1" if (new <= old - t) else "0")
PY
)
      if [[ "$VALUE_GATE_PASS" == "1" ]]; then
        cp "$VALUE_SAVE_TS" "$VALUE_CURRENT"
        {
          echo "val_mae=$VALUE_NEW_MAE"
          echo "source=$VALUE_SAVE_TS"
          echo "updated_at=$(date '+%Y-%m-%d %H:%M:%S')"
        } > "$VALUE_META_FILE"
        VALUE_GATE_STATUS="accepted"
        echo "[Round $ROUND] VALUE GATE: ACCEPTED (old_mae=$VALUE_OLD_MAE new_mae=$VALUE_NEW_MAE improve=$VALUE_GATE_DELTA)"
      else
        VALUE_GATE_STATUS="rejected"
        echo "[Round $ROUND] VALUE GATE: REJECTED (old_mae=$VALUE_OLD_MAE new_mae=$VALUE_NEW_MAE improve=$VALUE_GATE_DELTA)"
        echo "[Round $ROUND] value_current remains unchanged: $VALUE_CURRENT"
      fi
    fi
  else
    cp "$VALUE_SAVE_TS" "$VALUE_CURRENT"
    if [[ -n "$VALUE_NEW_MAE" ]]; then
      {
        echo "val_mae=$VALUE_NEW_MAE"
        echo "source=$VALUE_SAVE_TS"
        echo "updated_at=$(date '+%Y-%m-%d %H:%M:%S')"
      } > "$VALUE_META_FILE"
    fi
    VALUE_GATE_STATUS="disabled_promoted"
    echo "[Round $ROUND] VALUE GATE: DISABLED (promoted value checkpoint to current)"
  fi

  {
    echo ""
    echo "## Iteration Round $(printf '%02d' "$ROUND") - $(date '+%Y-%m-%d %H:%M:%S')"
    echo "- policy checkpoint: \`$POLICY_TS_CKPT\`"
    echo "- policy gate: $POLICY_GATE_STATUS"
    echo "- policy avg_reward before/after: ${POLICY_BEFORE_REWARD:-n/a} -> ${POLICY_AFTER_REWARD:-n/a} (delta=${POLICY_GATE_DELTA})"
    echo "- value checkpoint: \`$VALUE_SAVE_TS\`"
    echo "- value gate: $VALUE_GATE_STATUS"
    echo "- value val_mae old/new: ${VALUE_OLD_MAE:-n/a} -> ${VALUE_NEW_MAE:-n/a} (improve=${VALUE_GATE_DELTA})"
    echo "- policy training: episodes=$POLICY_EPISODES sims=$POLICY_SIMS updates=$POLICY_UPDATES eval_games=$POLICY_EVAL_GAMES"
    echo "- value training: episodes=$VALUE_EPISODES updates=$VALUE_UPDATES"
  } >> "$NOTES_FILE"

done

echo
echo "=== Iteration pipeline complete ==="
echo "Current aliases:"
echo "  $POLICY_CURRENT"
echo "  $VALUE_CURRENT"
echo "Review checkpoints in:"
echo "  $POLICY_DIR"
echo "  $VALUE_DIR"
