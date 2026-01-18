import numpy as np
import gameLogic

NUM_PLAYERS = 4
NUM_CARDS = 52
NUM_ACTIONS = 1695


def _one_hot(index, size):
    vec = np.zeros((size,), dtype=np.float32)
    if index is None:
        return vec
    if 0 <= index < size:
        vec[index] = 1.0
    return vec


def _card_rank_suit(card_id):
    # card_id in [1, 52]
    rank = int(np.ceil(card_id / 4.0)) - 1  # 0-12
    suit = (card_id - 1) % 4  # 0-3 (C, D, H, S)
    return rank, suit


def _card_mask(card_ids):
    mask = np.zeros((NUM_CARDS,), dtype=np.float32)
    for cid in card_ids:
        mask[int(cid) - 1] = 1.0
    return mask


def _played_mask(game):
    if hasattr(game, "cardsPlayed"):
        return np.clip(game.cardsPlayed.sum(axis=0), 0, 1).astype(np.float32)
    return np.zeros((NUM_CARDS,), dtype=np.float32)


def _prev_hand_features(game):
    prev_type = np.zeros((7,), dtype=np.float32)  # none, single, pair, straight, full, four, straight_flush
    prev_rank = np.zeros((13,), dtype=np.float32)
    prev_suit = np.zeros((4,), dtype=np.float32)
    prev_straight = np.zeros((10,), dtype=np.float32)

    prev_index = game.goIndex - 1
    if prev_index < 1 or prev_index not in game.handsPlayed:
        prev_type[0] = 1.0
        return prev_type, prev_rank, prev_suit, prev_straight

    prev_hand = game.handsPlayed[prev_index].hand
    n_cards = len(prev_hand)
    if n_cards == 1:
        prev_type[1] = 1.0
        rank, suit = _card_rank_suit(int(np.max(prev_hand)))
        prev_rank[rank] = 1.0
        prev_suit[suit] = 1.0
    elif n_cards == 2:
        prev_type[2] = 1.0
        rank, suit = _card_rank_suit(int(np.max(prev_hand)))
        prev_rank[rank] = 1.0
        prev_suit[suit] = 1.0
    elif n_cards == 5:
        if gameLogic.isStraightFlush(prev_hand):
            prev_type[6] = 1.0
            idx, high = gameLogic.straightRank(prev_hand)
            prev_straight[idx] = 1.0
            rank, suit = _card_rank_suit(int(high))
            prev_rank[rank] = 1.0
            prev_suit[suit] = 1.0
        elif gameLogic.isFourOfAKind(prev_hand):
            prev_type[5] = 1.0
            val = int(gameLogic.fourOfAKindValue(prev_hand))
            prev_rank[val - 1] = 1.0
        elif gameLogic.isFullHouse(prev_hand)[0]:
            prev_type[4] = 1.0
            val = int(gameLogic.isFullHouse(prev_hand)[1])
            prev_rank[val - 1] = 1.0
        else:
            prev_type[3] = 1.0
            straight_info = gameLogic.straightRank(prev_hand)
            if straight_info is not None:
                idx, high = straight_info
                prev_straight[idx] = 1.0
                rank, suit = _card_rank_suit(int(high))
                prev_rank[rank] = 1.0
                prev_suit[suit] = 1.0
    else:
        prev_type[0] = 1.0

    return prev_type, prev_rank, prev_suit, prev_straight


def encode_belief_input(game, perspective_player):
    my_hand = _card_mask(game.currentHands[perspective_player])
    played = _played_mask(game)
    remaining = np.array(
        [len(game.currentHands[i]) for i in range(1, 5)],
        dtype=np.float32,
    ) / 13.0
    current_player = _one_hot(game.playersGo - 1, NUM_PLAYERS)
    control = np.array([1.0 if game.control else 0.0], dtype=np.float32)
    must_play_club3 = np.array([1.0 if game.mustPlayClub3 else 0.0], dtype=np.float32)
    passed = np.array(
        [1.0 if game.passedThisRound[i] else 0.0 for i in range(1, 5)],
        dtype=np.float32,
    )
    last_player = _one_hot(game.lastPlayedPlayer - 1, NUM_PLAYERS)

    prev_type, prev_rank, prev_suit, prev_straight = _prev_hand_features(game)

    features = [
        my_hand,
        played,
        remaining,
        current_player,
        control,
        must_play_club3,
        passed,
        last_player,
        prev_type,
        prev_rank,
        prev_suit,
        prev_straight,
    ]
    return np.concatenate(features, axis=0).astype(np.float32)


def belief_input_dim():
    dummy = np.zeros((1,), dtype=np.int64)
    class _DummyGame:
        pass
    g = _DummyGame()
    g.currentHands = {1: dummy, 2: dummy, 3: dummy, 4: dummy}
    g.cardsPlayed = np.zeros((4, 52), dtype=np.int8)
    g.playersGo = 1
    g.control = 0
    g.mustPlayClub3 = False
    g.passedThisRound = {1: False, 2: False, 3: False, 4: False}
    g.lastPlayedPlayer = 1
    g.goIndex = 0
    g.handsPlayed = {}
    return int(encode_belief_input(g, 1).shape[0])


def belief_targets(game, perspective_player):
    played = _played_mask(game)
    my_hand = _card_mask(game.currentHands[perspective_player])
    mask = 1.0 - np.clip(played + my_hand, 0, 1)
    targets = np.full((NUM_CARDS,), -1, dtype=np.int64)

    for p, idx in zip([2, 3, 4], [0, 1, 2]):
        for cid in game.currentHands[p]:
            card_idx = int(cid) - 1
            if mask[card_idx] > 0:
                targets[card_idx] = idx
    return targets, mask.astype(np.float32)


def encode_policy_input(game, perspective_player, belief_probs):
    base = encode_belief_input(game, perspective_player)
    belief_flat = belief_probs.reshape(-1).astype(np.float32)
    return np.concatenate([base, belief_flat], axis=0)
