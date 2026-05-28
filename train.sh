#!/usr/bin/env bash
# Train AlphaBig2 using the new engine.
# Usage: ./train.sh [--iterations N] [--sims N] [--self-play N]
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/../../../.venv/bin/python3"

# Fallback: check parent venv path used by main repo
if [[ ! -f "$VENV" ]]; then
  VENV="/Users/shukaihu/Code_Project_Local/AlphaBig2-claude/.venv/bin/python3"
fi

if [[ ! -f "$VENV" ]]; then
  echo "ERROR: Python venv not found. Expected at:"
  echo "  $VENV"
  exit 1
fi

export PYTHONPATH="$SCRIPT_DIR"
export PYTHONNOUSERSITE=1

echo "Starting training with: $VENV"
echo "Checkpoints will be saved to: $SCRIPT_DIR/engine/checkpoints/"
echo ""

"$VENV" -m engine.trainer "$@"
