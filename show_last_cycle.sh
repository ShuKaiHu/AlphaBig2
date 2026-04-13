#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")" || exit 1

RUNS_DIR=${RUNS_DIR:-runs}
latest_meta="${RUNS_DIR}/latest.meta"

if [[ ! -f "$latest_meta" ]]; then
  echo "No runs found in $RUNS_DIR"
  exit 0
fi

echo "=== Latest Run Metadata ==="
cat "$latest_meta"

summary_file=$(awk -F= '/^SUMMARY_FILE=/{print $2}' "$latest_meta" | tail -n 1)
log_file=$(awk -F= '/^FULL_LOG_FILE=/{print $2}' "$latest_meta" | tail -n 1)

echo
if [[ -n "$summary_file" && -f "$summary_file" ]]; then
  echo "=== Summary ==="
  cat "$summary_file"
else
  echo "Summary file not found yet"
  if [[ -n "$log_file" && -f "$log_file" ]]; then
    echo
    echo "=== Latest Log Tail ==="
    tail -n 40 "$log_file"
  fi
fi

echo
if [[ -n "$log_file" ]]; then
  echo "Full log: $log_file"
fi
