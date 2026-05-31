import numpy as np

import enumerateOptions
import gameLogic

from ML_AB.actions import ACTION_DIM, NUM_CARDS


HISTORY_LEN = 196
CARD_FEAT_DIM = 16
HISTORY_FEAT_DIM = 64
GLOBAL_FEAT_DIM = 32


def _card_mask(card_ids):
    out = np.zeros((NUM_CARDS,), dtype=np.float32)
    for cid in card_ids:
        out[int(cid) - 1] = 1.0
    return out


def _played_mask(game):
    return np.clip(game.cardsPlayed.sum(axis=0), 0, 1).astype(np.float32)


def _rel_index(abs_player, perspective_player):
    return (int(abs_player) - int(perspective_player)) % 4


def _abs_from_rel(rel, perspective_player):
    return ((int(perspective_player) - 1 + int(rel)) % 4) + 1


def public_belief_prior(game, perspective_player):
    """Uniform public prior over unknown cards, weighted by opponent hand sizes."""
    played = _played_mask(game)
    mine = _card_mask(game.currentHands[perspective_player])
    unknown = 1.0 - np.clip(played + mine, 0, 1)
    rem = []
    for rel in (1, 2, 3):
        p = _abs_from_rel(rel, perspective_player)
        rem.append(float(len(game.currentHands[p])))
    total = max(float(np.sum(rem)), 1.0)
    probs = np.zeros((NUM_CARDS, 3), dtype=np.float32)
    for i, count in enumerate(rem):
        probs[:, i] = unknown * (count / total)
    return probs


def belief_targets(game, perspective_player):
    played = _played_mask(game)
    mine = _card_mask(game.currentHands[perspective_player])
    unknown = 1.0 - np.clip(played + mine, 0, 1)
    targets = np.full((NUM_CARDS,), -1, dtype=np.int64)
    for rel in (1, 2, 3):
        p = _abs_from_rel(rel, perspective_player)
        for cid in game.currentHands[p]:
            idx = int(cid) - 1
            if unknown[idx] > 0:
                targets[idx] = rel - 1
    return targets, unknown.astype(np.float32)


def _last_hand(game):
    if game.goIndex - 1 in game.handsPlayed:
        return game.handsPlayed[game.goIndex - 1].hand
    return None


def _hand_type_features(hand, control):
    out = np.zeros((8,), dtype=np.float32)
    if control:
        out[0] = 1.0  # any
        return out
    if hand is None or len(hand) == 0:
        out[1] = 1.0
        return out
    hand = np.array(hand, dtype=int)
    if hand.size == 1:
        out[2] = 1.0
    elif hand.size == 2:
        out[3] = 1.0
    elif hand.size == 5 and gameLogic.isStraightFlush(hand):
        out[7] = 1.0
    elif hand.size == 5 and gameLogic.isFourOfAKind(hand):
        out[6] = 1.0
    elif hand.size == 5 and gameLogic.isFullHouse(hand)[0]:
        out[5] = 1.0
    elif hand.size == 5:
        out[4] = 1.0
    else:
        out[1] = 1.0
    return out


def encode_game(game, perspective_player, belief_probs=None):
    if belief_probs is None:
        belief_probs = public_belief_prior(game, perspective_player)
    belief_probs = np.array(belief_probs, dtype=np.float32).reshape(NUM_CARDS, 3)

    mine = _card_mask(game.currentHands[perspective_player])
    played_by_abs = np.clip(game.cardsPlayed, 0, 1).astype(np.float32)
    played = np.clip(played_by_abs.sum(axis=0), 0, 1)
    unknown = 1.0 - np.clip(mine + played, 0, 1)

    card_feats = np.zeros((NUM_CARDS, CARD_FEAT_DIM), dtype=np.float32)
    for idx in range(NUM_CARDS):
        rank = idx // 4
        suit = idx % 4
        card_feats[idx, 0] = mine[idx]
        card_feats[idx, 1] = played[idx]
        card_feats[idx, 2] = unknown[idx]
        card_feats[idx, 3] = rank / 12.0
        card_feats[idx, 4] = suit / 3.0
        card_feats[idx, 5 + rank % 4] = 1.0
        card_feats[idx, 9 + suit] = 1.0
        card_feats[idx, 13:16] = belief_probs[idx] * unknown[idx]

    hist = np.zeros((HISTORY_LEN, HISTORY_FEAT_DIM), dtype=np.float32)
    history = list(getattr(game, "actionHistory", []))[-HISTORY_LEN:]
    start = HISTORY_LEN - len(history)
    for pos, entry in enumerate(history, start=start):
        player = entry.get("player")
        rel = _rel_index(player, perspective_player) if player is not None else 0
        hist[pos, rel] = 1.0
        hist[pos, 4] = 1.0 if entry.get("pass", False) else 0.0
        hist[pos, 5] = 1.0 if entry.get("forced_skip", False) else 0.0
        hist[pos, 6] = 1.0 if entry.get("control_break", False) else 0.0
        hand = entry.get("hand")
        if hand is not None:
            for cid in hand:
                hist[pos, 7 + int(cid) - 1] = 1.0
            hist[pos, 59] = float(len(hand)) / 5.0
        passed = entry.get("passed_snapshot")
        if passed is not None:
            for abs_p in range(1, 5):
                hist[pos, 60 + _rel_index(abs_p, perspective_player)] = float(passed[abs_p - 1])

    global_feats = np.zeros((GLOBAL_FEAT_DIM,), dtype=np.float32)
    global_feats[0] = 1.0 if game.playersGo == perspective_player else 0.0
    global_feats[1 + _rel_index(game.playersGo, perspective_player)] = 1.0
    global_feats[5] = 1.0 if game.control else 0.0
    global_feats[6] = 1.0 if game.mustPlayClub3 else 0.0
    global_feats[7 + _rel_index(game.lastPlayedPlayer, perspective_player)] = 1.0
    for rel in range(4):
        abs_p = _abs_from_rel(rel, perspective_player)
        global_feats[11 + rel] = float(len(game.currentHands[abs_p])) / 13.0
        global_feats[15 + rel] = 1.0 if game.passedThisRound[abs_p] else 0.0
    global_feats[19:27] = _hand_type_features(_last_hand(game), game.control)
    global_feats[27] = float(game.passCount) / 3.0
    global_feats[28] = float(len(getattr(game, "actionHistory", []))) / 64.0
    global_feats[29] = float(np.sum(unknown)) / 39.0
    global_feats[30] = float(np.sum(played)) / 52.0
    global_feats[31] = 1.0

    return {
        "card_feats": card_feats,
        "history_feats": hist,
        "global_feats": global_feats,
    }


def action_mask(game):
    return game.returnAvailableActions().astype(np.float32).reshape(ACTION_DIM)


def apply_action(game, action):
    if int(action) == enumerateOptions.passInd:
        game.updateGame(-1)
    else:
        opt, n = enumerateOptions.getOptionNC(int(action))
        game.updateGame(opt, n)
