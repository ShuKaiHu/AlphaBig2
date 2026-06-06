# AlphaBig2-codex

AlphaBig2-codex is the ML/model repository for the Big Two agent. The active
production line is `ML_AB`: transformer policy/value/belief model plus live
MCTS tooling.

The connector/browser automation code lives in the sibling repository:

- `/Users/shukaihu/Code_Project_Local/Big2VisionAgent-codex`
- GitHub: `ShuKaiHu/Big2VisionAgent-codex`

## Active Model Line

Use this checkpoint for online testing unless a candidate has passed the reward
gate and been promoted:

```text
ML_AB/models/big2_transformer_best.pt
```

Checkpoint files under `ML_AB/models/` are intentionally ignored by git. Store
shared checkpoints through GitHub Releases or Git LFS, not ordinary commits.

## Active Code Paths

- `ML_AB/models.py`: transformer network with policy, value, and belief heads.
- `ML_AB/online.py`: online recommendation entry point for public game states.
- `ML_AB/live_mcts.py`: one-second live MCTS with posterior particles.
- `ML_AB/train.py`: base self-play training.
- `ML_AB/train_search.py`: search/visit-target fine tuning.
- `ML_AB/train_live_corpus.py`: live reward/value calibration.
- `ML_AB/train_live_preferences.py`: targeted preference repair from reviewed
  online mistakes.
- `ML_AB/export_live_belief_dataset.py`: export belief labels from completed
  online rounds.
- `ML_AB/eval_reward_league.py`: reward-based evaluation gates.
- `ML_AB/eval_belief.py`: belief-head evaluation against public-prior baselines.

Legacy `ML_SKHU` experiments are preserved for reference, but they are not the
current online model pipeline. Old root-level ML_SKHU launch scripts were moved
to `legacy/ml_skhu_scripts/` so the root directory only shows active ML_AB
entry points.

## Online Test Command

Run this from `Big2VisionAgent-codex`, not this repository:

```bash
cd /Users/shukaihu/Code_Project_Local/Big2VisionAgent-codex

ALPHA_BIG2_CKPT=/Users/shukaihu/Code_Project_Local/AlphaBig2-codex/ML_AB/models/big2_transformer_best.pt \
ALPHA_BIG2_AGENT=mcts \
ALPHA_BIG2_MCTS_SECONDS=1.0 \
ALPHA_BIG2_MCTS_SELECTION=visits \
ALPHA_BIG2_MCTS_POSTERIOR_PARTICLES=24 \
ALPHA_BIG2_MCTS_HISTORY_WEIGHT=1.0 \
ALPHA_BIG2_MCTS_ROOT_WARMUP=12 \
ALPHA_BIG2_MCTS_ACTION_VALUE_FALLBACK_WEIGHT=0.0 \
BIG2_AGENT_COMMAND=/Users/shukaihu/Code_Project_Local/Big2VisionAgent-codex/alpha_big2_wrapper.py \
uv run big2-agent autoplay-agent --executor packet --games 3
```

## ML_AB Smoke Checks

```bash
EPISODES=20 BATCH=16 DMODEL=64 LAYERS=1 ./run_ml_ab_train.sh
GAMES=50 OPPONENT=random ./run_ml_ab_eval.sh
```

## Search Fine Tuning

```bash
INIT=ML_AB/models/big2_transformer_current.pt \
SAVE=ML_AB/models/big2_transformer_search_next.pt \
EPISODES=300 SIMS=16 BATCH=32 UPDATES=2 DEVICE=cpu \
./run_ml_ab_search_train.sh
```

## Live Data

Curated ML training data belongs under `ML_AB/data/`.

- `ML_AB/data/live_belief_dataset.jsonl`: tracked belief-label snapshot.
- `ML_AB/data/live_training_corpus.jsonl`: local reviewed decision corpus; this
  file is ignored by git unless explicitly published elsewhere.

The default paths can be overridden:

```bash
ALPHA_BIG2_LIVE_CORPUS=/path/to/live_training_corpus.jsonl
ALPHA_BIG2_LIVE_BELIEF_DATASET=/path/to/live_belief_dataset.jsonl
ALPHA_BIG2_VISION_AGENT_DIR=/path/to/Big2VisionAgent-codex
```

## Repository Boundary

This repository owns:

- game engine/rules used by ML,
- model architectures and training scripts,
- MCTS/search/evaluation code,
- curated ML datasets.

Big2VisionAgent-codex owns:

- browser login/session handling,
- WebSocket parsing,
- legal-action construction from live packets,
- executor behavior,
- online artifacts and review/export utilities.
