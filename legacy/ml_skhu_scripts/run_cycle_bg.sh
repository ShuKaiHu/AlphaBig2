#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")" || exit 1

RUNS_DIR=${RUNS_DIR:-runs}
mkdir -p "$RUNS_DIR"

if ! command -v screen >/dev/null 2>&1; then
  echo "screen not found"
  exit 1
fi

run_id=${RUN_ID:-full_cycle_$(date +%Y%m%d_%H%M%S)}
base_dir="$PWD"
log_file="${base_dir}/${RUNS_DIR}/${run_id}.log"
summary_file="${base_dir}/${RUNS_DIR}/${run_id}.summary"
meta_file="${base_dir}/${RUNS_DIR}/${run_id}.meta"
latest_meta="${base_dir}/${RUNS_DIR}/latest.meta"
launch_time=$(date '+%Y-%m-%d %H:%M:%S %z')
session_name="$run_id"
worker_script="${base_dir}/${RUNS_DIR}/${run_id}.worker.sh"

cat > "$worker_script" <<WORKER
#!/usr/bin/env bash
set +e
cd "$base_dir" || exit 1
export RUN_ID="$run_id"
export RUNS_DIR="$RUNS_DIR"
export FULL_LOG_FILE="$log_file"
export SUMMARY_FILE="$summary_file"
export META_FILE="$meta_file"
export LATEST_META_LINK="$latest_meta"
export DEVICE="${DEVICE:-cpu}"
export ROUNDS="${ROUNDS:-10}"
export POLICY_EPISODES="${POLICY_EPISODES:-200}"
export POLICY_UPDATES="${POLICY_UPDATES:-1}"
export POLICY_SIMS="${POLICY_SIMS:-120}"
export POLICY_EVAL_GAMES="${POLICY_EVAL_GAMES:-1000}"
export POLICY_LOG_EVERY="${POLICY_LOG_EVERY:-20}"
export POLICY_GATE_MIN_DELTA="${POLICY_GATE_MIN_DELTA:-0.3}"
export TARGET_TEMP_START="${TARGET_TEMP_START:-1.0}"
export TARGET_TEMP_END="${TARGET_TEMP_END:-0.5}"
export SKIP_VALUE="${SKIP_VALUE:-0}"
export VALUE_TRAIN_EVERY_N_ROUNDS="${VALUE_TRAIN_EVERY_N_ROUNDS:-5}"
export VALUE_EPISODES="${VALUE_EPISODES:-1000}"
export VALUE_LOG_EVERY="${VALUE_LOG_EVERY:-100}"
export SEL_GAMES="${SEL_GAMES:-1000}"
export FIN_GAMES="${FIN_GAMES:-2000}"
export TOPK="${TOPK:-3}"
export TAG1="${TAG1:-$(date +%Y%m%d)}"
export TAG2="${TAG2:-$(date -v+1d +%Y%m%d)}"
./run_full_cycle_and_select.sh
WORKER
printf 'rc=$?\n' >> "$worker_script"
cat >> "$worker_script" <<'WORKER'
if [[ ! -f "$SUMMARY_FILE" ]]; then
  {
    echo "=== Full Cycle Summary ==="
    echo "RUN_ID=$RUN_ID"
    echo "STATUS=failed"
    echo "LAST_STAGE=unknown"
    echo "FULL_LOG_FILE=$FULL_LOG_FILE"
    echo "META_FILE=$META_FILE"
    echo
    echo "=== Failure ==="
    echo "screen worker exited without summary (exit=$rc)"
    if [[ -f "$FULL_LOG_FILE" ]]; then
      echo
      echo "=== Last Log Lines ==="
      tail -n 40 "$FULL_LOG_FILE" || true
    fi
  } > "$SUMMARY_FILE"
fi
exit "$rc"
WORKER
chmod +x "$worker_script"

screen -dmS "$session_name" bash "$worker_script"
sleep 1
screen_pid=$( (screen -ls 2>/dev/null || true) | awk -v name="$session_name" '$0 ~ name {split($1,a,"."); print a[1]; exit}')

{
  echo "RUN_ID=$run_id"
  echo "SCREEN_SESSION=$session_name"
  if [[ -n "${screen_pid:-}" ]]; then
    echo "PID=$screen_pid"
  fi
  echo "LAUNCH_TIME=$launch_time"
  echo "DEVICE=${DEVICE:-cpu}"
  echo "ROUNDS=${ROUNDS:-10}"
  echo "POLICY_EPISODES=${POLICY_EPISODES:-200}"
  echo "POLICY_UPDATES=${POLICY_UPDATES:-1}"
  echo "POLICY_SIMS=${POLICY_SIMS:-120}"
  echo "POLICY_EVAL_GAMES=${POLICY_EVAL_GAMES:-1000}"
  echo "POLICY_LOG_EVERY=${POLICY_LOG_EVERY:-20}"
  echo "POLICY_GATE_MIN_DELTA=${POLICY_GATE_MIN_DELTA:-0.3}"
  echo "TARGET_TEMP_START=${TARGET_TEMP_START:-1.0}"
  echo "TARGET_TEMP_END=${TARGET_TEMP_END:-0.5}"
  echo "SKIP_VALUE=${SKIP_VALUE:-0}"
  echo "VALUE_TRAIN_EVERY_N_ROUNDS=${VALUE_TRAIN_EVERY_N_ROUNDS:-5}"
  echo "VALUE_EPISODES=${VALUE_EPISODES:-1000}"
  echo "VALUE_LOG_EVERY=${VALUE_LOG_EVERY:-100}"
  echo "SEL_GAMES=${SEL_GAMES:-1000}"
  echo "FIN_GAMES=${FIN_GAMES:-2000}"
  echo "TOPK=${TOPK:-3}"
  echo "TAG1=${TAG1:-$(date +%Y%m%d)}"
  echo "TAG2=${TAG2:-$(date -v+1d +%Y%m%d)}"
  echo "FULL_LOG_FILE=$log_file"
  echo "SUMMARY_FILE=$summary_file"
  echo "META_FILE=$meta_file"
  echo "WORKER_SCRIPT=$worker_script"
} > "$meta_file"

cp "$meta_file" "$latest_meta"

echo "Started background cycle"
echo "RUN_ID=$run_id"
echo "SCREEN_SESSION=$session_name"
if [[ -n "${screen_pid:-}" ]]; then
  echo "PID=$screen_pid"
fi
echo "LOG=$log_file"
echo "SUMMARY=$summary_file"
echo "META=$meta_file"
