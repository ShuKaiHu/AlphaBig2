import argparse
import math
import random

import numpy as np
import torch

import big2Game
import enumerateOptions

from ML_SKHU.value import ValueModel
from ML_SKHU.features import encode_policy_input


RANK_STR = ["3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A", "2"]
SUIT_STR = ["C", "D", "H", "S"]  # card_id % 4 -> 1:C 2:D 3:H 0:S


def _card_to_str(card_id):
    c = int(card_id)
    rank_idx = (c - 1) // 4
    suit_idx = (c - 1) % 4
    return f"{RANK_STR[rank_idx]}{SUIT_STR[suit_idx]}"


def _cards_to_str(cards):
    arr = [int(x) for x in np.array(cards).reshape(-1)]
    return "[" + " ".join(_card_to_str(c) for c in arr) + "]"


def _heuristic_action(game):
    avail = game.returnAvailableActions()
    valid = np.flatnonzero(avail == 1)
    if valid.size == 0:
        return enumerateOptions.passInd
    non_pass = valid[valid != enumerateOptions.passInd]
    if non_pass.size > 0:
        return int(np.min(non_pass))
    return int(enumerateOptions.passInd)


def _apply_action(game, action):
    if action == enumerateOptions.passInd:
        game.updateGame(-1)
    else:
        opt, n = enumerateOptions.getOptionNC(action)
        game.updateGame(opt, n)


def _oracle_belief_probs(game):
    probs = np.zeros((52, 3), dtype=np.float32)
    for abs_player, ch in [(2, 0), (3, 1), (4, 2)]:
        for cid in game.currentHands[abs_player]:
            probs[int(cid) - 1, ch] = 1.0
    return probs


def _value_input_dim():
    g = big2Game.big2Game()
    return int(encode_policy_input(g, 1, _oracle_belief_probs(g)).shape[0])


def _predict_value(model, game, player, device):
    x = encode_policy_input(game, player, _oracle_belief_probs(game))
    t = torch.from_numpy(x).float().unsqueeze(0).to(device)
    with torch.no_grad():
        pred = float(model(t).view(-1).cpu().numpy()[0])
    return pred


def _inv_tanh_scaled(v, value_scale):
    v = float(np.clip(v, -0.999999, 0.999999))
    return math.atanh(v) * float(value_scale)


def _last_hand_str(game):
    idx = game.goIndex - 1
    if idx in game.handsPlayed:
        hp = game.handsPlayed[idx]
        return f"P{hp.player} {_cards_to_str(hp.hand)}"
    return "None"


def _capture_state(game, player, turn_idx):
    return {
        "turn": int(turn_idx),
        "player": int(player),
        "playersGo": int(game.playersGo),
        "remaining": [int(len(game.currentHands[i])) for i in range(1, 5)],
        "control": int(game.control),
        "mustPlayClub3": bool(game.mustPlayClub3),
        "passed": [bool(game.passedThisRound[i]) for i in range(1, 5)],
        "last_player": int(game.lastPlayedPlayer),
        "last_hand": _last_hand_str(game),
        "hands": {i: np.array(game.currentHands[i]).copy() for i in range(1, 5)},
        "game_clone": game.clone(),
    }


def collect_endgame_samples(
    value_model,
    device="cpu",
    games=50,
    max_samples=8,
    total_cards_threshold=12,
    max_hand_threshold=5,
    seed=0,
):
    random.seed(seed)
    np.random.seed(seed)

    samples = []
    for gidx in range(1, games + 1):
        game = big2Game.big2Game()
        turn_idx = 0
        pending = []
        while not game.gameOver:
            turn_idx += 1
            p = game.playersGo

            rem = [len(game.currentHands[i]) for i in range(1, 5)]
            total_rem = sum(rem)
            max_rem = max(rem)
            if total_rem <= total_cards_threshold or max_rem <= max_hand_threshold:
                pred = _predict_value(value_model, game, p, device)
                snap = _capture_state(game, p, turn_idx)
                snap["game_index"] = gidx
                snap["total_remaining"] = int(total_rem)
                snap["pred_value"] = float(pred)
                pending.append(snap)

            a = _heuristic_action(game)
            _apply_action(game, a)

        # finalize pending samples with terminal reward from this heuristic continuation
        r1 = float(game.rewards[0])
        for s in pending:
            s["true_reward_p1"] = r1
            samples.append(s)

    if not samples:
        return []

    # Prefer late positions but sample diverse examples
    samples.sort(key=lambda s: (s["total_remaining"], max(s["remaining"])))
    if len(samples) <= max_samples:
        return samples

    chosen = []
    idxs = np.linspace(0, len(samples) - 1, num=max_samples, dtype=int)
    seen = set()
    for i in idxs:
        if int(i) not in seen:
            seen.add(int(i))
            chosen.append(samples[int(i)])
    while len(chosen) < max_samples:
        chosen.append(samples[random.randrange(len(samples))])
    return chosen[:max_samples]


def rank_samples_by_error(samples, value_scale, mode="abs_err"):
    ranked = []
    for s in samples:
        pred = float(s["pred_value"])
        true_r = float(s["true_reward_p1"])
        true_scaled = math.tanh(true_r / float(value_scale))
        err = pred - true_scaled
        sign_wrong = (pred > 0 and true_scaled < 0) or (pred < 0 and true_scaled > 0)
        s["true_scaled"] = float(true_scaled)
        s["err"] = float(err)
        s["abs_err"] = float(abs(err))
        s["sign_wrong"] = bool(sign_wrong)
        ranked.append(s)

    if mode == "abs_err":
        ranked.sort(key=lambda x: x["abs_err"], reverse=True)
    elif mode == "sign_wrong_then_abs":
        ranked.sort(key=lambda x: (0 if x["sign_wrong"] else 1, -x["abs_err"]))
    elif mode == "optimistic_mistakes":
        # cases where model is too optimistic (pred - true is large positive)
        ranked.sort(key=lambda x: x["err"], reverse=True)
    elif mode == "pessimistic_mistakes":
        ranked.sort(key=lambda x: x["err"])  # most negative first
    else:
        raise ValueError(f"unknown rank mode: {mode}")
    return ranked


def print_samples(samples, value_scale):
    if not samples:
        print("No endgame samples found. Try increasing --games or thresholds.")
        return

    for i, s in enumerate(samples, start=1):
        pred = float(s["pred_value"])
        true_r = float(s["true_reward_p1"])
        true_scaled = math.tanh(true_r / float(value_scale))
        pred_raw = _inv_tanh_scaled(pred, value_scale)
        err = pred - true_scaled
        print("=" * 72)
        print(
            f"sample={i} game={s['game_index']} turn={s['turn']} "
            f"leaf_player=P{s['player']} total_remaining={s['total_remaining']}"
        )
        print(
            f"remaining(P1..P4)={s['remaining']} control={s['control']} "
            f"last_player=P{s['last_player']} passed={s['passed']}"
        )
        print(f"last_hand={s['last_hand']}")
        print(
            f"value_pred_scaled={pred:+.4f} "
            f"value_pred_implied_reward≈{pred_raw:+.2f} "
            f"true_reward_p1={true_r:+.2f} true_scaled={true_scaled:+.4f} "
            f"err={err:+.4f}"
        )
        print("hands:")
        for p in range(1, 5):
            print(f"  P{p} {s['remaining'][p-1]} cards {_cards_to_str(s['hands'][p])}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--value-ckpt", required=True)
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--total-cards-threshold", type=int, default=12)
    parser.add_argument("--max-hand-threshold", type=int, default=5)
    parser.add_argument("--value-scale", type=float, default=15.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--rank-mode",
        choices=["abs_err", "sign_wrong_then_abs", "optimistic_mistakes", "pessimistic_mistakes"],
        default="abs_err",
        help="How to sort endgame samples before showing top-N.",
    )
    parser.add_argument(
        "--show-all-collected",
        action="store_true",
        help="Collect all candidate endgame samples and rank them before truncating to --samples.",
    )
    args = parser.parse_args()

    dim = _value_input_dim()
    model = ValueModel(dim, hidden_dim=512).to(args.device)
    model.load_state_dict(torch.load(args.value_ckpt, map_location=args.device))
    model.eval()

    samples = collect_endgame_samples(
        model,
        device=args.device,
        games=args.games,
        max_samples=(10**9 if args.show_all_collected else args.samples),
        total_cards_threshold=args.total_cards_threshold,
        max_hand_threshold=args.max_hand_threshold,
        seed=args.seed,
    )
    samples = rank_samples_by_error(samples, value_scale=args.value_scale, mode=args.rank_mode)
    samples = samples[: args.samples]
    print_samples(samples, value_scale=args.value_scale)


if __name__ == "__main__":
    main()
