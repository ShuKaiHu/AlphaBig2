# ML_AB framework

This is the rebuilt AlphaBig2 training stack. It is kept separate from
`ML_SKHU/` while it is being validated.

## Design

- Transformer state encoder over:
  - 52 card tokens
  - 64 public action-history tokens
  - one global game token
- Structured action encoder for all 14,739 fixed action indices.
- Single network with policy, value, and belief heads.
- Training uses public belief priors, not oracle opponent cards.
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
