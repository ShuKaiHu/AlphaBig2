#!/usr/bin/env bash
set -e
python3 -m ml.imitation_data --games 300 --mcts-simulations 80 --mcts-rollout-steps 150 --out ml/imitation_dataset.npz
python3 -m ml.train_imitation --data ml/imitation_dataset.npz --epochs 50 --save ml/imitation_policy.pt
python3 -m ml.eval_mcts --episodes 50 --simulations 80 --rollout-steps 150 --belief-model ml/belief_model_best.pt --belief-data ml/belief_dataset_mcts.npz --rollout-model ml/imitation_policy.pt
