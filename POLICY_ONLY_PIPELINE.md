## Policy-only reward-focused pipeline

### Overview
This thread of work replaces the earlier mixed belief/policy setup with a **straight reward optimization pipeline**. Each round of training:

1. Plays `500` games from the perspective of `player1`, initially against three heuristic players and, on subsequent rounds, against the winner of the previous round.  
2. Trains a new `PolicyValueModel` on that data using **only the value head** (no REINFORCE, no action loss) so that reward prediction is the single objective.  
3. Evaluates the newly trained model via `ML_SKHU/policy_only/eval_reward.py` (500 tests) and **keeps it only if its `avg_reward` exceeds the best reward so far**.

Selected models are saved under `policy_selected_runs/runN` and are the only checkpoints that survive the selection gate.

### Scripts

- `run_policy_reward_pipeline.sh` – automates 5 iterations: self-play → train → reward eval → selection → final `cross_eval`.  
- `policy_selfplay.py` – now accepts `player-actor`/`opp-actor` arguments so you can switch between heuristics, random players, or the current best model.  
- `ml_SKHU/policy_only/train_policy_only.py` – trains with value loss only; logging reports `loss/action/value` for inspection.  
- `ML_SKHU/policy_only/eval_reward.py` – simplifies evaluation to a single avg reward number (player1 vs heuristic).  
- `ML_SKHU/policy_only/cross_eval.py` – compares the selected checkpoints, printing win rates and reward triples per `(player1 run, opponent run)` cell. Each cell now lists `player1/player2/player3/player4` performance for easy inspection.

### Cross-eval interpretation

The `cross_eval` table shows rows = which model supplies players 2-4, columns = which model is player1. Each cell is formatted as `p1/p2/p3/p4` win rate or reward. The high `+0.49/+0.18/+0.16/+0.17` cell you highlighted is the scenario where `player1=run2` dominates when starting first, but the same run loses heavily when forced into the later positions, indicating positional volatility rather than a pipeline failure.

### Next steps on the other machine

1. `git pull` the repo to get the new scripts and documentation.  
2. Run `source .venv/bin/activate && ./run_policy_reward_pipeline.sh`.  
3. Inspect `policy_selected_runs/` and rerun `cross_eval` or `eval_policy_only.py` if you need more data.  
4. Use the `POLICY_ONLY_PIPELINE.md` doc as your guide for how the pipeline and cross-eval cells are structured.

If you need me to sketch charts, logs, or a CSV dump of round rewards, tell me and I’ll add them before you switch machines.
