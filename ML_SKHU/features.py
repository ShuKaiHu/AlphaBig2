import numpy as np
import gameLogic
import enumerateOptions

NUM_PLAYERS = 4
NUM_CARDS = 52
NUM_ACTIONS = enumerateOptions.passInd + 1
HISTORY_LEN = 100


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


def _card_index_from_rank_suit(rank_idx, suit_idx):
    # rank_idx: 0..12 for 3..2, suit_idx: 0..3 for C,D,H,S
    return rank_idx * 4 + suit_idx


def _played_mask(game):
    if hasattr(game, "cardsPlayed"):
        return np.clip(game.cardsPlayed.sum(axis=0), 0, 1).astype(np.float32)
    return np.zeros((NUM_CARDS,), dtype=np.float32)


def _played_by_player(game):
    if hasattr(game, "cardsPlayed"):
        return np.clip(game.cardsPlayed, 0, 1).astype(np.float32)
    return np.zeros((NUM_PLAYERS, NUM_CARDS), dtype=np.float32)


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


def _history_step_vector(entry):
    player = entry.get("player")
    hand = entry.get("hand")
    passed = bool(entry.get("pass", False))
    forced_skip = bool(entry.get("forced_skip", False))
    control_break = bool(entry.get("control_break", False))
    passed_snapshot = entry.get("passed_snapshot")

    player_vec = _one_hot(player - 1, NUM_PLAYERS) if player is not None else np.zeros((NUM_PLAYERS,), dtype=np.float32)
    pass_flag = np.array([1.0 if passed else 0.0], dtype=np.float32)
    forced_skip_flag = np.array([1.0 if forced_skip else 0.0], dtype=np.float32)
    control_break_flag = np.array([1.0 if control_break else 0.0], dtype=np.float32)
    played_cards_mask = _card_mask(hand) if hand is not None else np.zeros((NUM_CARDS,), dtype=np.float32)
    if passed_snapshot is None:
        passed_snapshot = np.zeros((NUM_PLAYERS,), dtype=np.float32)
    else:
        passed_snapshot = np.array(passed_snapshot, dtype=np.float32).reshape(NUM_PLAYERS,)

    return np.concatenate(
        [
            player_vec,          # 4
            pass_flag,           # 1
            forced_skip_flag,    # 1
            control_break_flag,  # 1
            played_cards_mask,   # 52
            passed_snapshot,     # 4
        ],
        axis=0,
    )


def _history_features(game, history_len=HISTORY_LEN):
    vectors = []
    if hasattr(game, "actionHistory") and game.actionHistory:
        history = list(game.actionHistory)
    else:
        history = []
    recent_history = history[-history_len:]
    pad_len = history_len - len(recent_history)
    step_dim = 63
    for _ in range(pad_len):
        vectors.append(np.zeros((step_dim,), dtype=np.float32))
    for entry in recent_history:
        vectors.append(_history_step_vector(entry))
    return np.concatenate(vectors, axis=0)


def encode_belief_input(game, perspective_player):
    self_hand = _card_mask(game.currentHands[perspective_player])
    played_by_player = _played_by_player(game)
    remaining = np.array([len(game.currentHands[i]) for i in range(1, 5)], dtype=np.float32) / 13.0
    current_player = _one_hot(game.playersGo - 1, NUM_PLAYERS)
    control = np.array([1.0 if game.control else 0.0], dtype=np.float32)
    must_play_club3 = np.array([1.0 if game.mustPlayClub3 else 0.0], dtype=np.float32)
    passed = np.array([1.0 if game.passedThisRound[i] else 0.0 for i in range(1, 5)], dtype=np.float32)
    last_player = _one_hot(game.lastPlayedPlayer - 1, NUM_PLAYERS)
    history = _history_features(game)

    return np.concatenate(
        [
            self_hand,                         # 52
            played_by_player.reshape(-1),      # 208
            remaining,                         # 4
            current_player,                    # 4
            control,                           # 1
            must_play_club3,                   # 1
            passed,                            # 4
            last_player,                       # 4
            history,                           # 100 * 63
        ],
        axis=0,
    ).astype(np.float32)


def _belief_probs_to_4ch(belief_probs, unknown_mask):
    # Accept [52,3] or [52,4], return [52,4] as [P2,P3,P4,Punknown].
    b = np.array(belief_probs, dtype=np.float32)
    if b.ndim != 2 or b.shape[0] != NUM_CARDS:
        raise ValueError(f"belief_probs must be [52,3] or [52,4], got {b.shape}")
    if b.shape[1] == 4:
        p = b.copy()
    elif b.shape[1] == 3:
        p = np.zeros((NUM_CARDS, 4), dtype=np.float32)
        p[:, :3] = b
        residual = 1.0 - np.clip(np.sum(b, axis=1), 0.0, 1.0)
        p[:, 3] = residual
    else:
        raise ValueError(f"belief_probs must be [52,3] or [52,4], got {b.shape}")

    # Known cards should not contribute to belief channels.
    p = p * unknown_mask.reshape(-1, 1)
    # Keep each row numerically bounded.
    p = np.clip(p, 0.0, 1.0)
    row_sum = np.sum(p, axis=1, keepdims=True)
    nz = row_sum.squeeze(-1) > 1e-8
    p[nz] = p[nz] / row_sum[nz]
    return p.astype(np.float32)


def _straight_rank_sequences():
    # Rank indices where 0..12 maps to 3..2.
    # A2345 (smallest), 23456 (largest), and middle runs.
    return [
        [11, 12, 0, 1, 2],   # A-2-3-4-5
        [12, 0, 1, 2, 3],    # 2-3-4-5-6
        [0, 1, 2, 3, 4],     # 3-4-5-6-7
        [1, 2, 3, 4, 5],     # 4-5-6-7-8
        [2, 3, 4, 5, 6],     # 5-6-7-8-9
        [3, 4, 5, 6, 7],     # 6-7-8-9-10
        [4, 5, 6, 7, 8],     # 7-8-9-10-J
        [5, 6, 7, 8, 9],     # 8-9-10-J-Q
        [6, 7, 8, 9, 10],    # 9-10-J-Q-K
        [7, 8, 9, 10, 11],   # 10-J-Q-K-A
    ]


def encode_belief_summary_v1(game, perspective_player, belief_probs):
    """
    Return:
      raw_belief_flat: [52*4] = [P2,P3,P4,Punknown] per card
      summary: [20]
    """
    played = _played_mask(game)
    my_hand = _card_mask(game.currentHands[perspective_player])
    unknown_mask = 1.0 - np.clip(played + my_hand, 0, 1)

    probs4 = _belief_probs_to_4ch(belief_probs, unknown_mask)
    opp_probs = probs4[:, :3]  # [52,3]

    # 1) expected remaining by player (3)
    expected_remaining = np.sum(opp_probs, axis=0)

    # 2) expected twos by player (3), card ids 49..52 => indices 48..51
    twos_idx = np.array([48, 49, 50, 51], dtype=np.int64)
    twos_expected = np.sum(opp_probs[twos_idx], axis=0)

    # 3) spade 2 probability by player (3), card id 52 => index 51
    spade2_prob = opp_probs[51]

    # 4) expected high cards by player (3), J,Q,K,A,2 => rank indices 8..12
    high_rank_indices = np.concatenate([np.arange(8, 13) * 4 + s for s in range(4)]).astype(np.int64)
    high_cards_expected = np.sum(opp_probs[high_rank_indices], axis=0)

    # 5) bomb probabilities by player (6): four-kind and straight-flush
    four_kind_probs = np.zeros((3,), dtype=np.float32)
    for p in range(3):
        best = 0.0
        for rank_idx in range(13):
            card_inds = [rank_idx * 4 + s for s in range(4)]
            val = float(np.prod(opp_probs[card_inds, p]))
            if val > best:
                best = val
        four_kind_probs[p] = best

    sf_sequences = _straight_rank_sequences()
    straight_flush_probs = np.zeros((3,), dtype=np.float32)
    for p in range(3):
        best = 0.0
        for suit_idx in range(4):
            for seq in sf_sequences:
                inds = [_card_index_from_rank_suit(r, suit_idx) for r in seq]
                val = float(np.prod(opp_probs[inds, p]))
                if val > best:
                    best = val
        straight_flush_probs[p] = best

    # 6) unknown stats (2)
    unknown_mass_total = float(np.sum(probs4[:, 3]))
    unknown_mass_ratio = unknown_mass_total / float(NUM_CARDS)

    summary = np.concatenate(
        [
            expected_remaining.astype(np.float32),     # 3
            twos_expected.astype(np.float32),          # 3
            spade2_prob.astype(np.float32),            # 3
            high_cards_expected.astype(np.float32),    # 3
            four_kind_probs.astype(np.float32),        # 3
            straight_flush_probs.astype(np.float32),   # 3
            np.array([unknown_mass_total, unknown_mass_ratio], dtype=np.float32),  # 2
        ],
        axis=0,
    )
    return probs4.reshape(-1).astype(np.float32), summary.astype(np.float32)


def encode_full_info_input(game, perspective_player):
    hands = []
    for i in range(1, 5):
        if i == perspective_player:
            hands.append(_card_mask(game.currentHands[i]))
        else:
            hands.append(np.zeros((NUM_CARDS,), dtype=np.float32))
    all_hands = np.concatenate(hands, axis=0)
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
    unknown_mask = np.zeros((52,), dtype=np.float32)
    for i in range(1, 5):
        if i == perspective_player:
            continue
        for cid in game.currentHands[i]:
            unknown_mask[int(cid) - 1] = 1.0

    features = [
        all_hands,
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
        unknown_mask,
    ]
    return np.concatenate(features, axis=0).astype(np.float32)


def encode_full_info_input_known(game, perspective_player):
    hands = []
    for i in range(1, 5):
        hands.append(_card_mask(game.currentHands[i]))
    all_hands = np.concatenate(hands, axis=0)
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
    unknown_mask = np.zeros((52,), dtype=np.float32)

    features = [
        all_hands,
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
        unknown_mask,
    ]
    return np.concatenate(features, axis=0).astype(np.float32)


def encode_value_input_fullinfo(game):
    # Value-only input: full hands + last hand type (+any) + pass flags.
    hands = []
    for i in range(1, 5):
        hands.append(_card_mask(game.currentHands[i]))
    all_hands = np.concatenate(hands, axis=0)

    prev_type, _, _, _ = _hand_features(
        game.handsPlayed[game.goIndex - 1].hand if game.goIndex - 1 in game.handsPlayed else None
    )
    # Add extra "any" slot as part of hand-type one-hot when control is with the current player.
    any_flag = np.array([1.0 if game.control else 0.0], dtype=np.float32)
    if game.control:
        prev_type = np.zeros_like(prev_type)
    hand_type = np.concatenate([prev_type, any_flag], axis=0)

    passed = np.array(
        [1.0 if game.passedThisRound[i] else 0.0 for i in range(1, 5)],
        dtype=np.float32,
    )

    current_player = _one_hot(game.playersGo - 1, NUM_PLAYERS)

    return np.concatenate([all_hands, hand_type, current_player, passed], axis=0).astype(np.float32)


def encode_full_info_with_belief(game, perspective_player, belief_probs, margin=0.8):
    my_hand = _card_mask(game.currentHands[perspective_player])
    played = _played_mask(game)
    unknown_mask = 1.0 - np.clip(played + my_hand, 0, 1)

    # belief_probs: [52,3] (p2, p3, p4). Use confidence gap to gate low-confidence cards.
    top2 = np.partition(belief_probs, -2, axis=1)[:, -2:]
    confidence = top2[:, 1] - top2[:, 0]
    conf_mask = (confidence >= margin).astype(np.float32)
    p2 = belief_probs[:, 0] * unknown_mask * conf_mask
    p3 = belief_probs[:, 1] * unknown_mask * conf_mask
    p4 = belief_probs[:, 2] * unknown_mask * conf_mask

    hands = []
    for i in range(1, 5):
        if i == perspective_player:
            hands.append(my_hand)
        elif i == 2:
            hands.append(p2.astype(np.float32))
        elif i == 3:
            hands.append(p3.astype(np.float32))
        else:
            hands.append(p4.astype(np.float32))
    all_hands = np.concatenate(hands, axis=0)

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
        all_hands,
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
        unknown_mask.astype(np.float32),
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
    raw_belief_flat, belief_summary = encode_belief_summary_v1(
        game, perspective_player, belief_probs
    )
    return np.concatenate([base, raw_belief_flat, belief_summary], axis=0).astype(np.float32)
