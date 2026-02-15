#!/usr/bin/env bash
set -euo pipefail

GAMES=500
EPOCHS=40
VALUE_WEIGHT=3.0
ROUNDS=5
ITER_DIR=policy_iter_runs
SELECTED_DIR=policy_selected_runs

mkdir -p "$ITER_DIR" "$SELECTED_DIR"

best_model=""
best_reward=-1e9
selected_count=0

for ROUND in $(seq 1 $ROUNDS); do
  TARGET_DIR="$ITER_DIR/run$ROUND"
  mkdir -p "$TARGET_DIR"

  echo "=== round $ROUND: self-play ==="
  if [[ $ROUND -eq 1 ]]; then
    python -m ML_SKHU.policy_only.policy_selfplay \
      --games "$GAMES" \
      --player-actor heuristic \
      --opp-actor heuristic \
      --out "$TARGET_DIR/policy_data.npz"
  else
    python -m ML_SKHU.policy_only.policy_selfplay \
      --games "$GAMES" \
      --player-actor model \
      --player-model "$best_model" \
      --opp-actor model \
      --opp-model "$best_model" \
      --out "$TARGET_DIR/policy_data.npz"
  fi

  echo "=== round $ROUND: train ==="
  python -m ML_SKHU.policy_only.train_policy_only \
    --data "$TARGET_DIR/policy_data.npz" \
    --epochs "$EPOCHS" \
    --value-weight "$VALUE_WEIGHT" \
    --action-weight 0.0 \
    --save "$TARGET_DIR/policy_only.pt"

  echo "=== round $ROUND: eval reward ==="
  reward=$(python -m ML_SKHU.policy_only.eval_reward \
    --policy-ckpt "$TARGET_DIR/policy_only.pt" \
    --games "$GAMES")

  echo "round $ROUND avg_reward=$reward"
  if (( $(echo "$reward > $best_reward" | bc -l) )); then
    echo "new best reward: $reward"
    best_reward="$reward"
    best_model="$TARGET_DIR/policy_only.pt"
    selected_count=$((selected_count + 1))
    mkdir -p "$SELECTED_DIR/run$selected_count"
    cp "$best_model" "$SELECTED_DIR/run$selected_count/policy_only.pt"
    echo "kept run $ROUND as selected run$selected_count"
  else
    echo "kept previous best reward $best_reward"
  fi
done

if [[ $selected_count -eq 0 ]]; then
  echo "no round improved reward; keeping last model"
  selected_count=1
  mkdir -p "$SELECTED_DIR/run1"
  cp "$best_model" "$SELECTED_DIR/run1/policy_only.pt"
fi

echo "=== final cross eval ==="
python -m ML_SKHU.policy_only.cross_eval \
  --base-dir "$SELECTED_DIR" \
  --runs "$selected_count" \
  --games "$GAMES"
