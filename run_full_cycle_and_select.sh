#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")" || exit 1

if [[ ! -f .venv/bin/activate ]]; then
  echo "Missing virtualenv: .venv/bin/activate"
  exit 1
fi

source .venv/bin/activate

DEVICE=${DEVICE:-cpu}
ROUNDS=${ROUNDS:-10}
POLICY_EPISODES=${POLICY_EPISODES:-200}
POLICY_UPDATES=${POLICY_UPDATES:-1}
POLICY_SIMS=${POLICY_SIMS:-120}
POLICY_EVAL_GAMES=${POLICY_EVAL_GAMES:-1000}
POLICY_LOG_EVERY=${POLICY_LOG_EVERY:-20}
POLICY_GATE_MIN_DELTA=${POLICY_GATE_MIN_DELTA:-0.3}
TARGET_TEMP_START=${TARGET_TEMP_START:-1.0}
TARGET_TEMP_END=${TARGET_TEMP_END:-0.5}
SKIP_VALUE=${SKIP_VALUE:-0}
VALUE_TRAIN_EVERY_N_ROUNDS=${VALUE_TRAIN_EVERY_N_ROUNDS:-5}
VALUE_EPISODES=${VALUE_EPISODES:-1000}
VALUE_LOG_EVERY=${VALUE_LOG_EVERY:-100}

SEL_GAMES=${SEL_GAMES:-1000}
FIN_GAMES=${FIN_GAMES:-2000}
TOPK=${TOPK:-3}

POLICY_DIR=${POLICY_DIR:-ML_SKHU/models/policy}
BELIEF_CKPT=${BELIEF_CKPT:-ML_SKHU/models/belief/belief_current.pt}
VALUE_CKPT=${VALUE_CKPT:-ML_SKHU/models/value/value_current.pt}

TAG1=${TAG1:-$(date +%Y%m%d)}
TAG2=${TAG2:-$(date -v+1d +%Y%m%d)}
RUN_ID=${RUN_ID:-manual_$(date +%Y%m%d_%H%M%S)}
RUNS_DIR=${RUNS_DIR:-runs}
FULL_LOG_FILE=${FULL_LOG_FILE:-}
SUMMARY_FILE=${SUMMARY_FILE:-}
META_FILE=${META_FILE:-}
LATEST_META_LINK=${LATEST_META_LINK:-${RUNS_DIR}/latest.meta}

mkdir -p "$RUNS_DIR"
if [[ -n "$FULL_LOG_FILE" ]]; then
  mkdir -p "$(dirname "$FULL_LOG_FILE")"
fi
if [[ -n "$SUMMARY_FILE" ]]; then
  mkdir -p "$(dirname "$SUMMARY_FILE")"
fi
if [[ -n "$META_FILE" ]]; then
  mkdir -p "$(dirname "$META_FILE")"
fi

CONFIG_LINES=(
  "RUN_ID=$RUN_ID"
  "DEVICE=$DEVICE"
  "ROUNDS=$ROUNDS"
  "POLICY_EPISODES=$POLICY_EPISODES"
  "POLICY_UPDATES=$POLICY_UPDATES"
  "POLICY_SIMS=$POLICY_SIMS"
  "POLICY_EVAL_GAMES=$POLICY_EVAL_GAMES"
  "POLICY_LOG_EVERY=$POLICY_LOG_EVERY"
  "POLICY_GATE_MIN_DELTA=$POLICY_GATE_MIN_DELTA"
  "TARGET_TEMP_START=$TARGET_TEMP_START"
  "TARGET_TEMP_END=$TARGET_TEMP_END"
  "SKIP_VALUE=$SKIP_VALUE"
  "VALUE_TRAIN_EVERY_N_ROUNDS=$VALUE_TRAIN_EVERY_N_ROUNDS"
  "VALUE_EPISODES=$VALUE_EPISODES"
  "VALUE_LOG_EVERY=$VALUE_LOG_EVERY"
  "SEL_GAMES=$SEL_GAMES"
  "FIN_GAMES=$FIN_GAMES"
  "TOPK=$TOPK"
  "TAG1=$TAG1"
  "TAG2=$TAG2"
)

print_config() {
  for line in "${CONFIG_LINES[@]}"; do
    echo "$line"
  done
}

summary_tmp=$(mktemp)
tmp_sel=$(mktemp)
tmp_fin=$(mktemp)
full_output_tmp=$(mktemp)
status="failed"
last_stage="startup"
best_ckpt=""
final_eval_output=""
failure_message=""

cleanup() {
  rm -f "$tmp_sel" "$tmp_fin" "$full_output_tmp" "$summary_tmp"
}
trap cleanup EXIT

write_summary() {
  {
    echo "=== Full Cycle Summary ==="
    print_config
    echo "STATUS=$status"
    echo "LAST_STAGE=$last_stage"
    if [[ -n "$FULL_LOG_FILE" ]]; then
      echo "FULL_LOG_FILE=$FULL_LOG_FILE"
    fi
    if [[ -n "$META_FILE" ]]; then
      echo "META_FILE=$META_FILE"
    fi
    echo
    echo "=== Training Config ==="
    echo "DEVICE=$DEVICE ROUNDS=$ROUNDS POLICY_EPISODES=$POLICY_EPISODES POLICY_SIMS=$POLICY_SIMS"
    echo "POLICY_EVAL_GAMES=$POLICY_EVAL_GAMES POLICY_GATE_MIN_DELTA=$POLICY_GATE_MIN_DELTA"
    echo "VALUE: skip=$SKIP_VALUE every_n_rounds=$VALUE_TRAIN_EVERY_N_ROUNDS episodes=$VALUE_EPISODES"
    echo "SELECT: sel_games=$SEL_GAMES fin_games=$FIN_GAMES topk=$TOPK tags=$TAG1,$TAG2"
    echo
    if grep -q "POLICY GATE\|VALUE GATE" "$full_output_tmp" 2>/dev/null; then
      echo "=== Gate Summary ==="
      grep "POLICY GATE\|VALUE GATE" "$full_output_tmp"
      echo
    fi
    if [[ -s "$tmp_sel" ]]; then
      echo "=== 初選 Top ${TOPK} ==="
      sort -gr "$tmp_sel" | head -n "$TOPK"
      echo
    fi
    if [[ -s "$tmp_fin" ]]; then
      echo "=== 決賽結果 ==="
      sort -gr "$tmp_fin"
      echo
    fi
    if [[ -n "$best_ckpt" ]]; then
      echo "Promoted -> ${POLICY_DIR}/policy_current.pt"
      echo "BEST_CHECKPOINT=$best_ckpt"
      echo
    fi
    if [[ -n "$final_eval_output" ]]; then
      echo "=== Final ${FIN_GAMES}局確認 ==="
      printf "%s\n" "$final_eval_output"
      echo
    fi
    if [[ -n "$failure_message" ]]; then
      echo "=== Failure ==="
      printf "%s\n" "$failure_message"
      echo
      echo "=== Last Log Lines ==="
      tail -n 40 "$full_output_tmp" 2>/dev/null || true
    fi
  } > "$summary_tmp"

  if [[ -n "$SUMMARY_FILE" ]]; then
    cp "$summary_tmp" "$SUMMARY_FILE"
  fi
}

on_error() {
  local exit_code=$?
  failure_message="Command failed during stage: ${last_stage} (exit=${exit_code})"
  status="failed"
  write_summary
  exit "$exit_code"
}
trap on_error ERR

existing_pid=""
existing_launch_time=""
if [[ -n "$META_FILE" && -f "$META_FILE" ]]; then
  existing_pid=$(awk -F= '/^PID=/{print $2}' "$META_FILE" | tail -n 1)
  existing_launch_time=$(awk -F= '/^LAUNCH_TIME=/{print $2}' "$META_FILE" | tail -n 1)
fi

if [[ -n "$META_FILE" ]]; then
  {
    echo "RUN_ID=$RUN_ID"
    if [[ -n "$existing_pid" ]]; then
      echo "PID=$existing_pid"
    fi
    if [[ -n "$existing_launch_time" ]]; then
      echo "LAUNCH_TIME=$existing_launch_time"
    fi
    echo "START_TIME=$(date '+%Y-%m-%d %H:%M:%S %z')"
    print_config
    if [[ -n "$FULL_LOG_FILE" ]]; then
      echo "FULL_LOG_FILE=$FULL_LOG_FILE"
    fi
    if [[ -n "$SUMMARY_FILE" ]]; then
      echo "SUMMARY_FILE=$SUMMARY_FILE"
    fi
  } > "$META_FILE"
  cp "$META_FILE" "$LATEST_META_LINK"
fi

if [[ -n "$FULL_LOG_FILE" ]]; then
  exec > >(tee -a "$full_output_tmp" "$FULL_LOG_FILE") 2>&1
else
  exec > >(tee -a "$full_output_tmp") 2>&1
fi

echo "=== Full Cycle Config ==="
echo "DEVICE=$DEVICE ROUNDS=$ROUNDS POLICY_EPISODES=$POLICY_EPISODES POLICY_SIMS=$POLICY_SIMS"
echo "POLICY_EVAL_GAMES=$POLICY_EVAL_GAMES POLICY_GATE_MIN_DELTA=$POLICY_GATE_MIN_DELTA"
echo "VALUE: skip=$SKIP_VALUE every_n_rounds=$VALUE_TRAIN_EVERY_N_ROUNDS episodes=$VALUE_EPISODES"
echo "SELECT: sel_games=$SEL_GAMES fin_games=$FIN_GAMES topk=$TOPK tags=$TAG1,$TAG2"
echo "RUN_ID=$RUN_ID"
echo

last_stage="train"
DEVICE=$DEVICE \
ROUNDS=$ROUNDS \
POLICY_EPISODES=$POLICY_EPISODES \
POLICY_UPDATES=$POLICY_UPDATES \
POLICY_SIMS=$POLICY_SIMS \
POLICY_EVAL_GAMES=$POLICY_EVAL_GAMES \
POLICY_LOG_EVERY=$POLICY_LOG_EVERY \
POLICY_GATE_MIN_DELTA=$POLICY_GATE_MIN_DELTA \
TARGET_TEMP_START=$TARGET_TEMP_START \
TARGET_TEMP_END=$TARGET_TEMP_END \
SKIP_VALUE=$SKIP_VALUE \
VALUE_TRAIN_EVERY_N_ROUNDS=$VALUE_TRAIN_EVERY_N_ROUNDS \
VALUE_EPISODES=$VALUE_EPISODES \
VALUE_LOG_EVERY=$VALUE_LOG_EVERY \
./run_iterate_policy_value.sh

candidates=()
for pattern in \
  "${POLICY_DIR}/policy_iter*_${TAG1}_*.pt" \
  "${POLICY_DIR}/policy_iter*_${TAG2}_*.pt"
do
  for ckpt in $pattern; do
    [[ -e "$ckpt" ]] && candidates+=("$ckpt")
  done
done

if [[ ${#candidates[@]} -eq 0 ]]; then
  failure_message="No checkpoints matched ${TAG1}/${TAG2}"
  exit 1
fi

last_stage="select"
echo "=== Stage 1: ${SEL_GAMES}局初選 ==="
for ckpt in "${candidates[@]}"; do
  out=$(python -m ML_SKHU.eval_three_models \
    --belief-ckpt "$BELIEF_CKPT" \
    --policy-ckpt "$ckpt" \
    --value-ckpt "$VALUE_CKPT" \
    --games "$SEL_GAMES" \
    --simulations "$POLICY_SIMS" \
    --opponent heuristic \
    --belief-mode oracle \
    --value-scale 15 \
    --device "$DEVICE")
  reward=$(echo "$out" | awk '/player1 win_rate:/ {print $NF}')
  printf "%s\t%s\n" "$reward" "$ckpt" | tee -a "$tmp_sel"
done

echo
echo "=== 初選 Top ${TOPK} ==="
top_list=$(sort -gr "$tmp_sel" | head -n "$TOPK")
echo "$top_list"

echo
echo "=== Stage 2: ${FIN_GAMES}局決賽 ==="
while IFS=$'\t' read -r _ ckpt; do
  [[ -z "${ckpt:-}" ]] && continue
  out=$(python -m ML_SKHU.eval_three_models \
    --belief-ckpt "$BELIEF_CKPT" \
    --policy-ckpt "$ckpt" \
    --value-ckpt "$VALUE_CKPT" \
    --games "$FIN_GAMES" \
    --simulations "$POLICY_SIMS" \
    --opponent heuristic \
    --belief-mode oracle \
    --value-scale 15 \
    --device "$DEVICE")
  reward=$(echo "$out" | awk '/player1 win_rate:/ {print $NF}')
  printf "%s\t%s\n" "$reward" "$ckpt" | tee -a "$tmp_fin"
done <<< "$top_list"

best_line=$(sort -gr "$tmp_fin" | head -n 1)
best_ckpt=$(echo "$best_line" | cut -f2-)

echo
echo "=== 決賽結果 ==="
sort -gr "$tmp_fin"

cp "$best_ckpt" "${POLICY_DIR}/policy_current.pt"
echo "Promoted -> ${POLICY_DIR}/policy_current.pt"

last_stage="verify"
echo
echo "=== Final ${FIN_GAMES}局確認 ==="
final_eval_output=$(python -m ML_SKHU.eval_three_models \
  --belief-ckpt "$BELIEF_CKPT" \
  --policy-ckpt "${POLICY_DIR}/policy_current.pt" \
  --value-ckpt "$VALUE_CKPT" \
  --games "$FIN_GAMES" \
  --simulations "$POLICY_SIMS" \
  --opponent heuristic \
  --belief-mode oracle \
  --value-scale 15 \
  --device "$DEVICE")
printf "%s\n" "$final_eval_output"

status="success"
last_stage="done"
write_summary
