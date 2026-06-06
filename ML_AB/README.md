# ML_AB framework

This is the active AlphaBig2 training stack. Legacy `ML_SKHU/` experiments are
kept for reference only; new online model work should use `ML_AB`.

## Design

- Transformer state encoder over:
  - 52 card tokens
  - 196 public action-history tokens
  - one global game token
- Structured action encoder for all 14,739 fixed action indices.
- Single network with policy, value, and belief heads.
- Live belief training can reconstruct hidden-card owner labels from complete
  online rounds. The target direction is serious belief-model imperfect-
  information MCTS: the belief head estimates each unseen card's owner
  distribution, and MCTS samples legal determinizations from that distribution.
- Checkpoints save config and metrics metadata, not just `state_dict`.

## Smoke commands

```bash
EPISODES=20 BATCH=16 DMODEL=64 LAYERS=1 ./run_ml_ab_train.sh
GAMES=50 OPPONENT=random ./run_ml_ab_eval.sh
```

## Longer bootstrap run

```bash
EPISODES=2000 BATCH=64 UPDATES=2 DMODEL=128 LAYERS=3 HEADS=4 ./run_ml_ab_train.sh
GAMES=1000 OPPONENT=heuristic ./run_ml_ab_eval.sh
```

## Search fine-tuning

After a bootstrap checkpoint exists, improve it with MCTS visit targets:

```bash
INIT=ML_AB/models/big2_transformer_current.pt \
SAVE=ML_AB/models/big2_transformer_search_next.pt \
EPISODES=300 SIMS=16 BATCH=32 UPDATES=2 DEVICE=cpu \
./run_ml_ab_search_train.sh
```

Current model line:

- `ML_AB/models/big2_transformer_best.pt`
- `ML_AB/models/big2_transformer_current.pt`
- History length: `196`
- Gate metric: player1 `avg_reward`, not win rate

The older 64-history checkpoints are no longer the active training line.

## Online recommendation adapter

The online client should construct an observation JSON with public data:

```json
{
  "my_hand": [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49],
  "perspective_player": 1,
  "current_player": 1,
  "opponent_counts": {"2": 13, "3": 13, "4": 13},
  "played_cards": [],
  "last_hand": null,
  "last_player": null,
  "control": true,
  "passed": {"1": false, "2": false, "3": false, "4": false}
}
```

Then run:

```bash
.venv/bin/python -m ML_AB.online --ckpt ML_AB/models/big2_transformer_current.pt --observation-json observation.json
```

## Live belief labels

Export reconstructed hidden-card owner labels from online artifacts:

```bash
uv run python -m ML_AB.export_live_belief_dataset \
  --save ML_AB/data/live_belief_dataset.jsonl
```

Evaluate a checkpoint against public-prior baselines, split by early/mid/late
phase:

```bash
uv run python -m ML_AB.eval_belief \
  --ckpt ML_AB/models/big2_transformer_best.pt \
  --dataset ML_AB/data/live_belief_dataset.jsonl
```

Until the belief head beats the public prior in these gates, live MCTS should
keep `ALPHA_BIG2_MCTS_BELIEF_BLEND` at `0.0` or a very small value.

## Live belief posterior

Live MCTS now samples hidden hands, replays public action history, and
reweights particles by action likelihood before running determinizations. This
lets pass/play evidence affect the hidden-hand posterior; for example, a
sample where the next player could beat a straight but passed is downweighted.

Useful online knobs:

```bash
ALPHA_BIG2_MCTS_POSTERIOR_PARTICLES=24
ALPHA_BIG2_MCTS_HISTORY_WEIGHT=1.0
ALPHA_BIG2_MCTS_BELIEF_BLEND=0.2
ALPHA_BIG2_MCTS_ROOT_WARMUP=12
ALPHA_BIG2_MCTS_ROOT_Q_MIN_ACTIONS=2
ALPHA_BIG2_MCTS_ROOT_Q_MIN_COVERAGE=0.7
ALPHA_BIG2_MCTS_ROOT_Q_MAX_REQUIRED=8
```

`model_debug.jsonl` records `mcts.belief_posterior`, including evidence counts,
effective sample size, and simple query probabilities such as whether the next
player likely holds any `2`.

## Live reward value training

Use live online corpus for value/reward calibration without imitating the
played actions:

```bash
uv run python -m ML_AB.train_live_corpus \
  --init ML_AB/models/candidate_belief_head_live_split.pt \
  --save ML_AB/models/candidate_value_head_live_mcts.pt \
  --metrics ML_AB/runs/live_mcts_value_head.jsonl \
  --agent-type mcts \
  --value-head-only \
  --policy-weight 0 \
  --value-weight 1 \
  --belief-weight 0 \
  --lr 5e-4 \
  --steps 500 \
  --batch-size 64 \
  --replay-repeat 8
```

`--value-head-only` keeps the policy, belief head, and shared encoder frozen.
The script reports train/eval value MAE so the checkpoint is judged by held-out
reward prediction, not by policy imitation accuracy.
