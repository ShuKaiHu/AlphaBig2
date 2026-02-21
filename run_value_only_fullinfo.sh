#!/usr/bin/env bash
set -euo pipefail

GAMES=2000
EPOCHS=100
ROUNDS=1
REWARD_GAMMA=0.9
ITER_DIR=value_iter_runs

mkdir -p "$ITER_DIR"

for ROUND in $(seq 1 $ROUNDS); do
  TARGET_DIR="$ITER_DIR/run$ROUND"
  mkdir -p "$TARGET_DIR"

  echo "=== round $ROUND: self-play (full-info value) ==="
  python -m ML_SKHU.value_only.value_selfplay \
    --games "$GAMES" \
    --actor heuristic \
    --reward-gamma "$REWARD_GAMMA" \
    --out "$TARGET_DIR/value_data.npz"

  echo "=== round $ROUND: train value ==="
  python -m ML_SKHU.value_only.train_value_only \
    --data "$TARGET_DIR/value_data.npz" \
    --epochs "$EPOCHS" \
    --lr 3e-5 \
    --freeze-policy \
    --save "$TARGET_DIR/value_only.pt" \
    --eval-every 10 \
    --eval-games 200 \
    --eval-actor heuristic

  echo "=== round $ROUND: eval value ==="
  python -m ML_SKHU.value_only.eval_value_only \
    --ckpt "$TARGET_DIR/value_only.pt" \
    --games "$GAMES" \
    --actor heuristic

done
