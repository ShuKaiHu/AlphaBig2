#!/usr/bin/env bash
set -euo pipefail

GAMES=1000
EPOCHS=10
VALUE_WEIGHT=1.0
ROUNDS=30
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
  SELFPLAY_MODEL_ARGS=()
  if [[ -n "$best_model" ]]; then
    SELFPLAY_MODEL_ARGS=(--player-model "$best_model")
  fi
  python -m ML_SKHU.policy_only.policy_selfplay \
    --games "$GAMES" \
    --player-actor model \
    ${SELFPLAY_MODEL_ARGS[@]+"${SELFPLAY_MODEL_ARGS[@]}"} \
    --opp-actor heuristic \
    --epsilon 0.08 \
    --reward-gamma 0.99 \
    --skip-forced-pass \
    --out "$TARGET_DIR/policy_data.npz"

  echo "=== round $ROUND: train ==="
  INIT_FROM=""
  if [[ -n "$best_model" ]]; then
    INIT_FROM="--init-from $best_model"
  fi
  python -m ML_SKHU.policy_only.train_policy_only \
    --data "$TARGET_DIR/policy_data.npz" \
    --epochs "$EPOCHS" \
    --value-weight "$VALUE_WEIGHT" \
    --action-weight 1.0 \
    --entropy-weight 0.1 \
    --ppo-clip 0.2 \
    --early-stop-patience 3 \
    --eval-games 100 \
    --eval-every 5 \
    --save "$TARGET_DIR/policy_only.pt" \
    $INIT_FROM

  echo "=== round $ROUND: eval reward ==="
  reward=$(python -m ML_SKHU.policy_only.eval_reward \
    --policy-ckpt "$TARGET_DIR/policy_only.pt" \
    --games "$GAMES")

  echo "round $ROUND avg_reward=$reward"
  if (( $(echo "$reward > $best_reward" | bc -l) )); then
    echo "new best reward: $reward"
    best_reward="$reward"
    best_model="$TARGET_DIR/policy_only.pt"
  else
    echo "kept previous best reward $best_reward"
  fi

  mkdir -p "$SELECTED_DIR/run$ROUND"
  cp "$TARGET_DIR/policy_only.pt" "$SELECTED_DIR/run$ROUND/policy_only.pt"
done

echo "=== final eval vs heuristic (all runs) ==="
for ROUND in $(seq 1 $ROUNDS); do
  run_ckpt="${SELECTED_DIR}/run${ROUND}/policy_only.pt"
  if [[ -f "${run_ckpt:-}" ]]; then
    score=$(python -m ML_SKHU.policy_only.eval_reward \
      --policy-ckpt "$run_ckpt" \
      --games "$GAMES")
    echo "run $ROUND avg_reward_vs_heur=$score"
  else
    echo "run $ROUND missing checkpoint"
  fi
done
