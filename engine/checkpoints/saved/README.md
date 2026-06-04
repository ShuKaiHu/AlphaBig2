# Saved AlphaBig2 deployment checkpoints

Pure `model_state` state-dicts (no optimizer) for `engine.model.Big2Net`.
Both use the 4-dim multi-player value head (max^n MCTS, the core 4-player fix).

## `baseline_v7.1.pt`  — 302-dim features (NO dominance)
- The original clean self-play run (sims=50, 50 ep/iter, cosine LR), iter ~414.
- Offline: MCTS-80 avg_score ≈ **+7.1** vs heuristic; imperfect-info single-deep determinization ≈ +12.
- Online vs real humans: ≈ break-even (avg ~+0.6, 22% go-out, 16% disaster rate).
- **Run with `BIG2_DOMINANCE` UNSET** (STATIC_DIM=302).

## `v6_dominance_deploy.pt`  — 306-dim features (Tier-A dominance)
- Same recipe + opt-in dominance features (BIG2_DOMINANCE=1), full 500 iters.
- Offline: MCTS-80 avg_score ≈ +5.8 vs heuristic (≈ baseline on average).
- Online vs real humans (79 games): avg ≈ +0.7 (≈ baseline) BUT distribution shifted
  the right way — **go-out 22%→33%, disaster rate 16%→8%, lower variance**
  (suggestive ~1.5σ, consistent with dominance's "will I get over-trumped" purpose).
- **MUST run with `BIG2_DOMINANCE=1`** (STATIC_DIM=306) — otherwise the wrapper's
  feature-dim guard will refuse to load it (434 vs 430 input mismatch).

## Loading
```python
import torch
from engine.model import Big2Net
m = Big2Net()                       # auto-sizes input from features.STATIC_DIM
m.load_state_dict(torch.load(PATH, map_location="cpu"))
m.eval()
```
For v6, set `BIG2_DOMINANCE=1` in the environment BEFORE importing engine.features.

Current online deployment = v6 (copied into engine/checkpoints/best.pt).
To revert to baseline: copy baseline_v7.1.pt → best.pt and run WITHOUT BIG2_DOMINANCE.
