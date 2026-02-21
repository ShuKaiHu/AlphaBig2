#!/usr/bin/env bash
set -euo pipefail

EPISODES=300
SIMULATIONS=50
BATCH_SIZE=32
UPDATES_PER_EPISODE=3
POLICY_WEIGHT=1.0
OPPONENT=heuristic
EVAL_GAMES=200
MODEL_DIR="ML_SKHU/models/${OPPONENT}"
FULLINFO_MODEL="policy_best/policy_only_fullinfo_vs_heuristic.pt"
BELIEF_CKPT="policy_best/belief_only_margin0.8_accknown_20260219_211436.pt"

echo "=== MCTS train (belief + policy_value) ==="
python -m ML_SKHU.train \
  --episodes "$EPISODES" \
  --simulations "$SIMULATIONS" \
  --batch-size "$BATCH_SIZE" \
  --updates-per-episode "$UPDATES_PER_EPISODE" \
  --policy-weight "$POLICY_WEIGHT" \
  --opponent "$OPPONENT" \
  --belief-ckpt "$BELIEF_CKPT"

echo "=== MCTS eval vs heuristic ==="
python -m ML_SKHU.eval \
  --games "$EVAL_GAMES" \
  --simulations "$SIMULATIONS" \
  --belief-ckpt "$BELIEF_CKPT" \
  --policy-ckpt "${MODEL_DIR}/policy_value.pt" \
  --opponent heuristic

if [[ -f "$FULLINFO_MODEL" ]]; then
  echo "=== Full-info upper bound (policy-only) ==="
  python -m ML_SKHU.policy_only.eval_reward \
    --policy-ckpt "$FULLINFO_MODEL" \
    --full-info-known \
    --games "$EVAL_GAMES" \
    --opponent heuristic
else
  echo "=== Full-info upper bound skipped (missing $FULLINFO_MODEL) ==="
fi
