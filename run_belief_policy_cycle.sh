#!/usr/bin/env bash
set -euo pipefail

# Cycle: train belief (policy fixed) -> eval -> train policy_value (belief fixed) -> eval
CYCLES=3
BELIEF_EPISODES=1000
POLICY_EPISODES=20
SIMULATIONS=25
BATCH_SIZE=64
UPDATES_PER_EPISODE=5
POLICY_WEIGHT=1.0
OPPONENT=heuristic
EVAL_GAMES=200
MODEL_DIR="ML_SKHU/models/${OPPONENT}"
BELIEF_MARGIN=0.9
LOG_EVERY=100
EVAL_EVERY=100

# Fixed checkpoints
BELIEF_CKPT="${MODEL_DIR}/belief.pt"
POLICY_ONLY_CKPT=""
BELIEF_DECAY=0.5

for CYCLE in $(seq 1 $CYCLES); do
  echo "=== Cycle $CYCLE-1: train belief (policy fixed) ==="
  python -m ML_SKHU.train \
    --episodes "$BELIEF_EPISODES" \
    --simulations 0 \
    --batch-size "$BATCH_SIZE" \
    --updates-per-episode "$UPDATES_PER_EPISODE" \
    --policy-weight "$POLICY_WEIGHT" \
    --opponent "$OPPONENT" \
    --train-belief \
    --log-every "$LOG_EVERY" \
    --eval-every "$EVAL_EVERY" \
    --eval-games "$EVAL_GAMES" \
    --belief-ckpt "$BELIEF_CKPT" \
    ${POLICY_ONLY_CKPT:+--policy-only-ckpt "$POLICY_ONLY_CKPT"} \
    --policy-only-player 1 \
    --belief-decay "$BELIEF_DECAY"

  echo "=== Cycle $CYCLE-2: eval reward (belief stage) ==="
  python -m ML_SKHU.eval \
    --games "$EVAL_GAMES" \
    --simulations 0 \
    --belief-ckpt "$BELIEF_CKPT" \
    ${POLICY_ONLY_CKPT:+--policy-only-ckpt "$POLICY_ONLY_CKPT"} \
    --opponent heuristic \
    --belief-margin "$BELIEF_MARGIN"

  echo "=== Cycle $CYCLE-3: train policy_value (belief fixed) ==="
  python -m ML_SKHU.train \
    --episodes "$POLICY_EPISODES" \
    --simulations "$SIMULATIONS" \
    --batch-size "$BATCH_SIZE" \
    --updates-per-episode "$UPDATES_PER_EPISODE" \
    --policy-weight "$POLICY_WEIGHT" \
    --opponent "$OPPONENT" \
    --train-policy \
    --log-every "$LOG_EVERY" \
    --eval-every "$EVAL_EVERY" \
    --eval-games "$EVAL_GAMES" \
    --belief-ckpt "$BELIEF_CKPT"

  echo "=== Cycle $CYCLE-4: eval reward (policy stage) ==="
  python -m ML_SKHU.eval \
    --games "$EVAL_GAMES" \
    --simulations "$SIMULATIONS" \
    --belief-ckpt "$BELIEF_CKPT" \
    --policy-ckpt "${MODEL_DIR}/policy_value.pt" \
    --opponent heuristic \
    --belief-margin "$BELIEF_MARGIN"
done
