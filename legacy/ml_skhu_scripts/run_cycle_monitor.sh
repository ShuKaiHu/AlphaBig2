#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")" || exit 1

RUNS_DIR=${RUNS_DIR:-runs}
LATEST_META="${RUNS_DIR}/latest.meta"
CHECK_INTERVAL_SECONDS=${CHECK_INTERVAL_SECONDS:-600}
STALE_LOG_SECONDS=${STALE_LOG_SECONDS:-1800}
MONITOR_LOG="${RUNS_DIR}/cycle_monitor.log"

mkdir -p "$RUNS_DIR"
touch "$MONITOR_LOG"

log_msg() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %z')" "$*" | tee -a "$MONITOR_LOG"
}

meta_value() {
  local key=$1
  local file=$2
  awk -F= -v key="$key" '$1 == key {print substr($0, length(key) + 2)}' "$file" | tail -n 1
}

screen_session_alive() {
  local session=$1
  [[ -n "$session" ]] || return 1
  screen -ls 2>/dev/null | grep -q "[[:space:]]*[0-9][0-9]*\\.${session}[[:space:]]"
}

log_is_stale() {
  local file=$1
  local now
  local mtime
  [[ -f "$file" ]] || return 0
  now=$(date +%s)
  mtime=$(stat -f %m "$file")
  (( now - mtime > STALE_LOG_SECONDS ))
}

restart_cycle() {
  local meta_file=$1
  local old_run_id=""
  old_run_id=$(meta_value RUN_ID "$meta_file" || true)

  local -a env_args=()
  local key
  for key in \
    DEVICE ROUNDS POLICY_EPISODES POLICY_UPDATES POLICY_SIMS POLICY_EVAL_GAMES POLICY_LOG_EVERY \
    POLICY_GATE_MIN_DELTA TARGET_TEMP_START TARGET_TEMP_END SKIP_VALUE VALUE_TRAIN_EVERY_N_ROUNDS \
    VALUE_EPISODES VALUE_LOG_EVERY SEL_GAMES FIN_GAMES TOPK TAG1 TAG2
  do
    local value=""
    value=$(meta_value "$key" "$meta_file" || true)
    if [[ -n "$value" ]]; then
      env_args+=("${key}=${value}")
    fi
  done

  log_msg "run ${old_run_id:-unknown} not operating; restarting"
  if [[ ${#env_args[@]} -gt 0 ]]; then
    env "${env_args[@]}" ./run_cycle_bg.sh >> "$MONITOR_LOG" 2>&1
  else
    ./run_cycle_bg.sh >> "$MONITOR_LOG" 2>&1
  fi
  local new_run_id=""
  if [[ -f "$LATEST_META" ]]; then
    new_run_id=$(meta_value RUN_ID "$LATEST_META" || true)
  fi
  log_msg "restart launched new run ${new_run_id:-unknown}"
}

log_msg "cycle monitor started interval=${CHECK_INTERVAL_SECONDS}s stale=${STALE_LOG_SECONDS}s"

while true; do
  if [[ ! -f "$LATEST_META" ]]; then
    log_msg "latest meta missing; starting new background cycle"
    ./run_cycle_bg.sh >> "$MONITOR_LOG" 2>&1
    sleep "$CHECK_INTERVAL_SECONDS"
    continue
  fi

  run_id=$(meta_value RUN_ID "$LATEST_META" || true)
  screen_session=$(meta_value SCREEN_SESSION "$LATEST_META" || true)
  summary_file=$(meta_value SUMMARY_FILE "$LATEST_META" || true)
  log_file=$(meta_value FULL_LOG_FILE "$LATEST_META" || true)

  if [[ -n "$summary_file" && -f "$summary_file" ]]; then
    log_msg "run ${run_id:-unknown} already has summary; monitor exiting"
    exit 0
  fi

  if screen_session_alive "$screen_session"; then
    log_msg "run ${run_id:-unknown} healthy; session=${screen_session}"
  else
    log_msg "run ${run_id:-unknown} screen session missing"
    restart_cycle "$LATEST_META"
    sleep "$CHECK_INTERVAL_SECONDS"
    continue
  fi

  if [[ -n "$log_file" ]] && log_is_stale "$log_file"; then
    log_msg "run ${run_id:-unknown} log stale for > ${STALE_LOG_SECONDS}s"
    restart_cycle "$LATEST_META"
    sleep "$CHECK_INTERVAL_SECONDS"
    continue
  fi

  sleep "$CHECK_INTERVAL_SECONDS"
done
