#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")" || exit 1
source "$(pwd)/scripts/ensure_repo_venv.sh"

EPISODES=1000
SIMULATIONS=0
BATCH_SIZE=64
UPDATES_PER_EPISODE=5
OPPONENT=heuristic
MODEL_DIR="ML_SKHU/models/${OPPONENT}"
EVAL_GAMES=200
POLICY_ONLY_CKPT=""

echo "=== Belief-only train (heuristic self-play, no MCTS) ==="
if [[ -n "$POLICY_ONLY_CKPT" && -f "$POLICY_ONLY_CKPT" ]]; then
  python -m ML_SKHU.train \
    --train-belief \
    --episodes "$EPISODES" \
    --simulations "$SIMULATIONS" \
    --batch-size "$BATCH_SIZE" \
    --updates-per-episode "$UPDATES_PER_EPISODE" \
    --policy-weight 0.0 \
    --opponent "$OPPONENT" \
    --belief-margin 0.9 \
    --log-every 100 \
    --eval-every 100 \
    --policy-only-ckpt "$POLICY_ONLY_CKPT"
else
  python -m ML_SKHU.train \
    --train-belief \
    --episodes "$EPISODES" \
    --simulations "$SIMULATIONS" \
    --batch-size "$BATCH_SIZE" \
    --updates-per-episode "$UPDATES_PER_EPISODE" \
    --policy-weight 0.0 \
    --opponent "$OPPONENT" \
    --belief-margin 0.9 \
    --log-every 100 \
    --eval-every 100
fi

echo "=== Belief eval (vs heuristic, using policy-only) ==="
if [[ -n "$POLICY_ONLY_CKPT" && -f "$POLICY_ONLY_CKPT" ]]; then
  python -m ML_SKHU.eval \
    --games "$EVAL_GAMES" \
    --simulations "$SIMULATIONS" \
    --belief-ckpt "${MODEL_DIR}/belief.pt" \
    --opponent heuristic \
    --belief-margin 0.8 \
    --policy-only-ckpt "$POLICY_ONLY_CKPT"
else
  python -m ML_SKHU.eval \
    --games "$EVAL_GAMES" \
    --simulations "$SIMULATIONS" \
    --belief-ckpt "${MODEL_DIR}/belief.pt" \
    --opponent heuristic \
    --belief-margin 0.8
fi
