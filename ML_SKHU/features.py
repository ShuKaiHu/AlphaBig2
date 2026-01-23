import numpy as np
import gameLogic

NUM_PLAYERS = 4
NUM_CARDS = 52
NUM_ACTIONS = 1695
HISTORY_LEN = 300


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


def _hand_features(hand):
    hand_type = np.zeros((7,), dtype=np.float32)  # none, single, pair, straight, full, four, straight_flush
    rank = np.zeros((13,), dtype=np.float32)
    suit = np.zeros((4,), dtype=np.float32)
    straight = np.zeros((10,), dtype=np.float32)

    if hand is None or len(hand) == 0:
        hand_type[0] = 1.0
        return hand_type, rank, suit, straight

    n_cards = len(hand)
    if n_cards == 1:
        hand_type[1] = 1.0
        r, s = _card_rank_suit(int(np.max(hand)))
        rank[r] = 1.0
        suit[s] = 1.0
    elif n_cards == 2:
        hand_type[2] = 1.0
        r, s = _card_rank_suit(int(np.max(hand)))
        rank[r] = 1.0
        suit[s] = 1.0
    elif n_cards == 5:
        if gameLogic.isStraightFlush(hand):
            hand_type[6] = 1.0
            idx, high = gameLogic.straightRank(hand)
            straight[idx] = 1.0
            r, s = _card_rank_suit(int(high))
            rank[r] = 1.0
            suit[s] = 1.0
        elif gameLogic.isFourOfAKind(hand):
            hand_type[5] = 1.0
            val = int(gameLogic.fourOfAKindValue(hand))
            rank[val - 1] = 1.0
        elif gameLogic.isFullHouse(hand)[0]:
            hand_type[4] = 1.0
            val = int(gameLogic.isFullHouse(hand)[1])
            rank[val - 1] = 1.0
        else:
            hand_type[3] = 1.0
            straight_info = gameLogic.straightRank(hand)
            if straight_info is not None:
                idx, high = straight_info
                straight[idx] = 1.0
                r, s = _card_rank_suit(int(high))
                rank[r] = 1.0
                suit[s] = 1.0
    else:
        hand_type[0] = 1.0

    return hand_type, rank, suit, straight


def _hand_action_vector(hand, player, passed):
    player_vec = _one_hot(player - 1, NUM_PLAYERS) if player is not None else np.zeros((NUM_PLAYERS,), dtype=np.float32)
    n_cards = 0 if hand is None else len(hand)
    n_cards_vec = np.zeros((4,), dtype=np.float32)  # pass/none, single, pair, five
    if n_cards == 1:
        n_cards_vec[1] = 1.0
    elif n_cards == 2:
        n_cards_vec[2] = 1.0
    elif n_cards == 5:
        n_cards_vec[3] = 1.0
    else:
        n_cards_vec[0] = 1.0
    pass_flag = np.array([1.0 if passed else 0.0], dtype=np.float32)
    hand_type, rank, suit, straight = _hand_features(hand)
    return np.concatenate([player_vec, n_cards_vec, pass_flag, hand_type, rank, suit, straight], axis=0)


def _history_features(game, history_len=HISTORY_LEN):
    vectors = []
    if hasattr(game, "actionHistory") and game.actionHistory:
        history = list(reversed(game.actionHistory))
    else:
        history = []
    for i in range(history_len):
        if i < len(history):
            entry = history[i]
            vectors.append(_hand_action_vector(entry["hand"], entry["player"], entry["pass"]))
        else:
            vectors.append(np.zeros((43,), dtype=np.float32))
    return np.concatenate(vectors, axis=0)


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

    history = _history_features(game)
    prev_type, prev_rank, prev_suit, prev_straight = _hand_features(
        game.handsPlayed[game.goIndex - 1].hand if game.goIndex - 1 in game.handsPlayed else None
    )

    features = [
        my_hand,
        played,
        remaining,
        current_player,
        control,
        must_play_club3,
        passed,
        last_player,
        history,
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
