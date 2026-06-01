"""Imperfect-information evaluation harness.

This is the measurement keystone for determinization work. Unlike eval_reward.py
(which lets MCTS see the TRUE hands = perfect info), here the model plays as it
would ONLINE: it does NOT see opponents' hands. At each of its turns it:

  1. computes the unknown cards (= opponents' real cards, by elimination) and
     each opponent's hand size,
  2. samples N "determinizations" — full hand assignments of the unknown cards
     to the 3 opponents (uniform, or belief-weighted),
  3. runs MCTS on each determinized clone, sums visit counts,
  4. plays argmax over the summed visits on the TRUE game.

Opponents play the heuristic. Reports avg_score (reward) for the model seat.

Usage:
  python eval_determinization.py CKPT --games 80 --dets 1 --sims 80          # current online (single uniform)
  python eval_determinization.py CKPT --games 80 --dets 4 --sims 20          # multi-det, equal total compute
  python eval_determinization.py CKPT --games 80 --dets 4 --sims 20 --belief # belief-guided multi-det
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from engine.model import Big2Net
from engine.env import Big2Env, PASS_IDX
from engine.mcts import MCTS
from engine.features import encode_static, encode_history_steps
from engine.self_play import _heuristic_action


def load_model(path):
    ck = torch.load(path, map_location="cpu")
    m = Big2Net()
    m.load_state_dict(ck["model_state"] if "model_state" in ck else ck)
    m.eval()
    return m


def belief_probs(model, game, player):
    static = encode_static(game, player)
    history = encode_history_steps(game)
    sf = torch.FloatTensor(static).unsqueeze(0)
    hs = torch.FloatTensor(history).unsqueeze(0)
    with torch.no_grad():
        _, _, bl = model.forward(sf, hs, None)
    return torch.sigmoid(bl).squeeze(0).numpy().reshape(3, 52)  # opp_idx × card


def sample_assignment(unknown, counts, belief=None, rng=None):
    """Assign unknown cards to the 3 opponents respecting `counts`.
    counts: dict {2:n2,3:n3,4:n4}. belief: (3,52) or None (uniform).
    Returns dict {2:list,3:list,4:list}."""
    rng = rng or np.random
    others = [2, 3, 4]
    order = list(unknown)
    rng.shuffle(order)
    caps = dict(counts)
    out = {2: [], 3: [], 4: []}
    for card in order:
        avail = [o for o in others if caps[o] > 0]
        if not avail:
            break
        if belief is not None:
            w = np.array([max(belief[others.index(o)][card - 1], 1e-6) for o in avail])
            w = w / w.sum()
            owner = avail[rng.choice(len(avail), p=w)]
        else:
            owner = avail[rng.choice(len(avail))]
        out[owner].append(card)
        caps[owner] -= 1
    return {o: np.sort(np.array(v, dtype=np.int64)) for o, v in out.items()}


def determinized_action(model, true_env, n_dets, sims, use_belief):
    game = true_env.game
    me = 1
    my = set(int(c) for c in game.currentHands[me])
    all_hands = set()
    for p in range(1, 5):
        all_hands |= set(int(c) for c in game.currentHands[p])
    unknown = sorted(all_hands - my)
    counts = {p: int(game.currentHands[p].size) for p in [2, 3, 4]}
    if len(unknown) == 0 or sum(counts.values()) == 0:
        # nothing hidden / terminal-ish; fall back to a single search on truth
        mcts = MCTS(model, n_simulations=sims)
        _, visits = mcts.run(true_env, temperature=1.0)
        return int(np.argmax(visits))

    belief = belief_probs(model, game, me) if use_belief else None
    agg = None
    for _ in range(n_dets):
        assign = sample_assignment(unknown, counts, belief=belief)
        d = true_env.clone()
        for p in [2, 3, 4]:
            d.game.currentHands[p] = assign[p]
        mcts = MCTS(model, n_simulations=sims)
        _, visits = mcts.run(d, temperature=1.0)
        agg = visits if agg is None else agg + visits
    return int(np.argmax(agg))


def run(path, games, n_dets, sims, use_belief, seed):
    model = load_model(path)
    np.random.seed(seed); torch.manual_seed(seed)
    scores, places = [], {1: 0, 2: 0, 3: 0, 4: 0}
    env = Big2Env()
    for _ in range(games):
        env.reset()
        while not env.done:
            p = env.current_player
            if p == 1:
                a = determinized_action(model, env, n_dets, sims, use_belief)
            else:
                a = _heuristic_action(env)
            rewards, done = env.step(a)
            if done:
                sc = float(rewards[0])
                scores.append(sc)
                # placement by reward rank (1 = best)
                r = rewards
                place = 1 + int(sum(1 for x in r if x > r[0]))
                places[place] = places.get(place, 0) + 1
                break
    scores = np.array(scores)
    mode = f"dets={n_dets} sims={sims} {'BELIEF' if use_belief else 'uniform'}"
    print(f"{mode:<34} avg_score={scores.mean():+.3f} ± {scores.std()/np.sqrt(len(scores)):.3f}"
          f"  1st={places[1]/games:.1%}  places={places}")
    return scores.mean()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--games", type=int, default=80)
    ap.add_argument("--dets", type=int, default=1)
    ap.add_argument("--sims", type=int, default=80)
    ap.add_argument("--belief", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    print(f"checkpoint: {a.ckpt}  (imperfect-info eval, opponents=heuristic)")
    run(a.ckpt, a.games, a.dets, a.sims, a.belief, a.seed)
