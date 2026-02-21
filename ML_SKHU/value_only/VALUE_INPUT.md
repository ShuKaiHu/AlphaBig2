# Value-Only Full-Info Input (v1)

This document describes the input vector used by the value-only model.

## Source
- `ML_SKHU/features.py` → `encode_value_input_fullinfo(game)`

## Feature Layout
1. **All players' hands (full info)**
   - 4 players × 52 cards = **208 dims**
   - Order: P1, P2, P3, P4 (each is a 52-dim card mask)

2. **Previous hand type (one-hot) + Any**
   - 7 base types from `_hand_features`:
     - none, single, pair, straight, full house, four of a kind, straight flush
   - +1 extra **"any"** slot when the current player has control
   - Total = **8 dims**
   - If `game.control` is True: base 7 are zeroed and **any=1**

3. **Current player (one-hot)**
   - P1..P4 = **4 dims**

4. **Passed this round**
   - P1..P4 = **4 dims**

## Total Dimension
`208 + 8 + 4 + 4 = 224`

## Notes
- This input is intended **only for value-only training**.
- Policy is not trained in this pipeline.
