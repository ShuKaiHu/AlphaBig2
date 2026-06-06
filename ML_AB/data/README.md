# ML_AB Data

This directory holds curated ML datasets that belong with the model/training
repo.

- `live_belief_dataset.jsonl`: tracked snapshot of reconstructed hidden-card
  belief labels from completed online rounds.
- `live_training_corpus.jsonl`: local decision/reward corpus generated from
  online artifacts. It can grow quickly and is ignored by git by default.

The scripts also accept explicit paths:

```bash
ALPHA_BIG2_LIVE_CORPUS=/path/to/live_training_corpus.jsonl
ALPHA_BIG2_LIVE_BELIEF_DATASET=/path/to/live_belief_dataset.jsonl
```
