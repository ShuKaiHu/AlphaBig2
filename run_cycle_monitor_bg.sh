#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")" || exit 1

RUNS_DIR=${RUNS_DIR:-runs}
mkdir -p "$RUNS_DIR"

if ! command -v screen >/dev/null 2>&1; then
  echo "screen not found"
  exit 1
fi

session_name=${MONITOR_SESSION_NAME:-cycle_monitor}
monitor_log="$PWD/${RUNS_DIR}/cycle_monitor.log"
monitor_meta="$PWD/${RUNS_DIR}/cycle_monitor.meta"

existing=$( (screen -ls 2>/dev/null || true) | awk -v name="$session_name" '$0 ~ "\."name"([[:space:]]|$)" {print $1; exit}')
if [[ -n "$existing" ]]; then
  echo "Monitor already running: $existing"
  echo "LOG=$monitor_log"
  exit 0
fi

screen -dmS "$session_name" bash -lc "cd '$PWD' && ./run_cycle_monitor.sh"
sleep 1
screen_pid=$( (screen -ls 2>/dev/null || true) | awk -v name="$session_name" '$0 ~ "\."name"([[:space:]]|$)" {split($1,a,"."); print a[1]; exit}')

{
  echo "SCREEN_SESSION=$session_name"
  if [[ -n "${screen_pid:-}" ]]; then
    echo "PID=$screen_pid"
  fi
  echo "START_TIME=$(date '+%Y-%m-%d %H:%M:%S %z')"
  echo "LOG=$monitor_log"
} > "$monitor_meta"

echo "Started cycle monitor"
echo "SCREEN_SESSION=$session_name"
if [[ -n "${screen_pid:-}" ]]; then
  echo "PID=$screen_pid"
fi
echo "LOG=$monitor_log"
echo "META=$monitor_meta"
