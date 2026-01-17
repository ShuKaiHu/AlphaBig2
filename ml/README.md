# Big2 local training (minimal)

This is a small, clean training scaffold that reuses the original game
environment but avoids the TensorFlow/baselines stack.

What you get:
- `env.py`: wrapper around the Big2 game with action masks
- `policy.py`: small MLP policy + value network (PyTorch)
- `train_reinforce.py`: minimal REINFORCE self-play trainer
- `train_ppo.py`: PPO self-play trainer with value head
- `eval.py`: quick evaluation against random or self-play

The training here is intentionally simple. It is a good starting point for
experimenting with rewards, feature engineering, and better RL algorithms.

Example usage:
- `python -m ml.train_ppo --episodes 200 --n-envs 4`
- `python -m ml.eval --model ml/models/ppo_ep200.pt --mode vs-baseline --baseline random`

Baselines:
- `random`: random valid action
- `min-index`: smallest valid action index (rough proxy for conservative play)
- `max-index`: largest valid action index (rough proxy for aggressive play)
