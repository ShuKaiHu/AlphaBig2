#!/bin/bash
# Training script for heuristic opponent
# Usage: bash train_heuristic.sh

cd "$(dirname "$0")" || exit

# Activate virtual environment
source .venv/bin/activate

# Train heuristic opponent with same hyperparameters as random
echo "Starting training for heuristic opponent..."
python -m ML_SKHU.train \
  --episodes 500 \
  --simulations 50 \
  --opponent heuristic \
  --device cpu \
  --batch-size 64 \
  --policy-weight 3.0 \
  --assume-yes

echo "Training complete. Models saved to ML_SKHU/models/heuristic/"
