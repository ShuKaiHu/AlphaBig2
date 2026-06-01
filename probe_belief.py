"""Measure whether the belief head actually predicts opponents' hidden cards.

For many mid-game self-play states we compare the belief head's per-opponent
card probabilities against the TRUE opponent hands and report:
  - precision@k : of the k cards the belief ranks highest for an opponent
                  (k = that opponent's true hand size), how many are correct
  - base rate   : k / (#unknown cards) — what random guessing would score
  - lift        : precision@k / base_rate  (1.0 = useless, >1 = informative)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from engine.model import Big2Net
from engine.env import Big2Env
from engine.features import encode_static, encode_history_steps
from engine.self_play import _heuristic_action

CKPT = sys.argv[1] if len(sys.argv) > 1 else "engine/checkpoints/latest.pt"
N_STATES = int(sys.argv[2]) if len(sys.argv) > 2 else 400

ck = torch.load(CKPT, map_location="cpu")
model = Big2Net()
state = ck["model_state"] if "model_state" in ck else ck
model.load_state_dict(state)
model.eval()

np.random.seed(0); torch.manual_seed(0)


def belief_probs(game, player):
    static = encode_static(game, player)
    history = encode_history_steps(game)
    sf = torch.FloatTensor(static).unsqueeze(0)
    hs = torch.FloatTensor(history).unsqueeze(0)
    with torch.no_grad():
        _, _, belief_logits = model.forward(sf, hs, None)
    probs = torch.sigmoid(belief_logits).squeeze(0).numpy()  # (156,)
    return probs.reshape(3, 52)  # opp_idx (others ascending) × card


# Bucket by game phase, measured by how many cards are still UNKNOWN
# (i.e. still in opponents' hands; opponents start with 39 total).
#   early: >27 unknown   mid: 15-27   late: <15
buckets = {
    "early (>27 unknown)": {"prec": [], "base": []},
    "mid (15-27 unknown)": {"prec": [], "base": []},
    "late (<15 unknown)":  {"prec": [], "base": []},
}


def _bucket(n_unknown):
    if n_unknown > 27:
        return "early (>27 unknown)"
    if n_unknown >= 15:
        return "mid (15-27 unknown)"
    return "late (<15 unknown)"


states_used = 0

for ep in range(80):
    env = Big2Env(); env.reset()
    steps = 0
    while not env.done and steps < 60:
        player = env.current_player
        game = env.game
        # Probe at EVERY player-1 turn (all phases) so we can bucket by phase.
        if player == 1:
            my = set(int(c) for c in game.currentHands[1])
            all_hands = set()
            for pp in range(1, 5):
                all_hands |= set(int(c) for c in game.currentHands[pp])
            unknown = sorted(all_hands - my)
            if len(unknown) >= 3:
                bp = belief_probs(game, 1)  # (3,52)
                others = [p for p in range(1, 5) if p != 1]  # [2,3,4]
                bk = _bucket(len(unknown))
                for oi, opp in enumerate(others):
                    true_hand = set(int(c) for c in game.currentHands[opp])
                    true_in_unknown = true_hand & set(unknown)
                    k = len(true_in_unknown)
                    if k == 0:
                        continue
                    scores = [(bp[oi][c - 1], c) for c in unknown]
                    scores.sort(reverse=True)
                    topk = set(c for _, c in scores[:k])
                    prec = len(topk & true_in_unknown) / k
                    base = k / len(unknown)
                    buckets[bk]["prec"].append(prec)
                    buckets[bk]["base"].append(base)
                    states_used += 1
        action = _heuristic_action(env)
        env.step(action)
        steps += 1

print(f"checkpoint: {CKPT}")
print(f"total opponent-state samples: {states_used}")
print(f"{'phase':<22}{'n':>7}{'prec@k':>9}{'base':>8}{'lift':>8}")
for name, d in buckets.items():
    if not d["prec"]:
        print(f"{name:<22}{0:>7}{'—':>9}")
        continue
    p = np.mean(d["prec"]); b = np.mean(d["base"])
    print(f"{name:<22}{len(d['prec']):>7}{p:>9.3f}{b:>8.3f}{p/b:>7.2f}x")
print("\nlift 1.0 = useless, >1.3 = clearly informative. "
      "Hypothesis: lift should rise sharply in the late phase.")
