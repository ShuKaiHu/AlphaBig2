#!/bin/bash
# Full training and evaluation pipeline
# Tasks: train random, train heuristic, eval random, eval heuristic

set -e  # Exit on error

cd "$(dirname "$0")" || exit

source .venv/bin/activate

echo "=========================================="
echo "AlphaBig2 Training & Evaluation Pipeline"
echo "=========================================="

# Task 1: Train random opponent
echo ""
echo "TASK 1: Training for random opponent..."
echo "=========================================="
python -m ML_SKHU.train \
  --episodes 500 \
  --simulations 50 \
  --opponent random \
  --device cpu \
  --batch-size 64 \
  --policy-weight 3.0 \
  --assume-yes
echo "✓ Random opponent training complete"
echo "  Models saved to: ML_SKHU/models/random/"

# Task 2: Train heuristic opponent
echo ""
echo "TASK 2: Training for heuristic opponent..."
echo "=========================================="
python -m ML_SKHU.train \
  --episodes 500 \
  --simulations 50 \
  --opponent heuristic \
  --device cpu \
  --batch-size 64 \
  --policy-weight 3.0 \
  --assume-yes
echo "✓ Heuristic opponent training complete"
echo "  Models saved to: ML_SKHU/models/heuristic/"

# Task 3: Evaluate random opponent model (and save if best)
echo ""
echo "TASK 3: Evaluating random opponent model..."
echo "=========================================="
python eval_and_save_best.py \
  --opponent random \
  --belief-ckpt ML_SKHU/models/random/belief.pt \
  --policy-ckpt ML_SKHU/models/random/policy_value.pt \
  --games 100 \
  --simulations 25
echo "✓ Random opponent evaluation & best-model check complete"

# Task 4: Evaluate heuristic opponent model (and save if best)
echo ""
echo "TASK 4: Evaluating heuristic opponent model..."
echo "=========================================="
python eval_and_save_best.py \
  --opponent heuristic \
  --belief-ckpt ML_SKHU/models/heuristic/belief.pt \
  --policy-ckpt ML_SKHU/models/heuristic/policy_value.pt \
  --games 100 \
  --simulations 25
echo "✓ Heuristic opponent evaluation & best-model check complete"

echo ""
echo "=========================================="
echo "✓ All tasks completed successfully!"
echo "=========================================="
echo ""
echo "Model locations:"
echo "  Random:    ML_SKHU/models/random/"
echo "  Heuristic: ML_SKHU/models/heuristic/"
echo ""
echo "Best models (based on player1 avg_reward):"
echo "  Random best:    ML_SKHU/models/best_random/"
echo "  Heuristic best: ML_SKHU/models/best_heuristic/"
