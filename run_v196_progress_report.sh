#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")" || exit 1
source "$(pwd)/scripts/ensure_repo_venv.sh"

AUTOMATION_DIR=${AUTOMATION_DIR:-$HOME/.codex/automations/alphabig2-v196-progress-report}
MEMORY_FILE=${MEMORY_FILE:-$AUTOMATION_DIR/memory.md}
AS_OF=${1:-now}

python scripts/generate_v196_progress_report.py \
  --repo-root "$(pwd)" \
  --automation-dir "$AUTOMATION_DIR" \
  --memory "$MEMORY_FILE" \
  --as-of "$AS_OF" \
  --output auto
