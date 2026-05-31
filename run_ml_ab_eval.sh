#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")" || exit 1
source "$(pwd)/scripts/ensure_repo_venv.sh"

CKPT=${CKPT:-ML_AB/models/big2_transformer_current.pt}
GAMES=${GAMES:-500}
OPPONENT=${OPPONENT:-heuristic}
DEVICE=${DEVICE:-cpu}
SEED=${SEED:-1}
AGENT=${AGENT:-model}
CONTROL_FIVE_BONUS=${CONTROL_FIVE_BONUS:-1.2}
CARD_COUNT_BONUS=${CARD_COUNT_BONUS:-0.12}
FINISH_BONUS=${FINISH_BONUS:-3.0}
URGENT_OPPONENT_COUNT=${URGENT_OPPONENT_COUNT:-3}
URGENT_FIVE_BONUS=${URGENT_FIVE_BONUS:-1.0}
PRESERVE_FIVE_CARD_PENALTY=${PRESERVE_FIVE_CARD_PENALTY:-0.25}

python -m ML_AB.eval \
  --ckpt "$CKPT" \
  --games "$GAMES" \
  --opponent "$OPPONENT" \
  --device "$DEVICE" \
  --seed "$SEED" \
  --agent "$AGENT" \
  --control-five-bonus "$CONTROL_FIVE_BONUS" \
  --card-count-bonus "$CARD_COUNT_BONUS" \
  --finish-bonus "$FINISH_BONUS" \
  --urgent-opponent-count "$URGENT_OPPONENT_COUNT" \
  --urgent-five-bonus "$URGENT_FIVE_BONUS" \
  --preserve-five-card-penalty "$PRESERVE_FIVE_CARD_PENALTY"
