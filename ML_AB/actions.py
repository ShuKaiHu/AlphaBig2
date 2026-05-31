import functools

import numpy as np
import torch

import enumerateOptions
import gameLogic


ACTION_DIM = enumerateOptions.passInd + 1
NUM_CARDS = 52


def action_cards(action):
    if int(action) == enumerateOptions.passInd:
        return []
    cards, _n = enumerateOptions.getOptionNC(int(action))
    return [int(c) for c in cards]


def action_to_string(action):
    cards = action_cards(action)
    if not cards:
        return "pass"
    return " ".join(card_label(c) for c in cards)


def action_combo_type(action):
    cards = np.array(action_cards(action), dtype=int)
    if cards.size == 0:
        return "pass"
    if cards.size == 1:
        return "single"
    if cards.size == 2:
        return "pair"
    if cards.size == 5:
        if gameLogic.isStraightFlush(cards.copy()):
            return "straight_flush"
        if gameLogic.isFourOfAKind(cards.copy()):
            return "four_of_a_kind"
        if gameLogic.isFullHouse(cards.copy())[0]:
            return "full_house"
        if gameLogic.isFlush(cards.copy()):
            return "flush"
        if gameLogic.isStraight(cards.copy()):
            return "straight"
        return "five_card"
    return f"{cards.size}_cards"


def card_label(card_id):
    ranks = ["3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A", "2"]
    suits = ["C", "D", "H", "S"]
    idx = int(card_id) - 1
    return f"{ranks[idx // 4]}{suits[idx % 4]}"


def _rank_index(card_id):
    return (int(card_id) - 1) // 4


def _suit_index(card_id):
    return (int(card_id) - 1) % 4


FIVE_CARD_SUBTYPES = ["straight", "flush", "full_house", "four_of_a_kind", "straight_flush"]
ACTION_FEAT_DIM = 81


def build_action_features():
    """Return [ACTION_DIM, ACTION_FEAT_DIM] structural features for every action index."""
    feats = np.zeros((ACTION_DIM, ACTION_FEAT_DIM), dtype=np.float32)
    for action in range(ACTION_DIM):
        cards = action_cards(action)
        offset = 0
        # Card mask: 52.
        for cid in cards:
            feats[action, offset + int(cid) - 1] = 1.0
        offset += 52

        # Action kind: pass, single, pair, five-card.
        if not cards:
            feats[action, offset + 0] = 1.0
        elif len(cards) == 1:
            feats[action, offset + 1] = 1.0
        elif len(cards) == 2:
            feats[action, offset + 2] = 1.0
        else:
            feats[action, offset + 3] = 1.0
        offset += 4

        # Five-card subtype: straight, flush, full house, four of a kind, straight flush.
        combo_type = action_combo_type(action)
        if combo_type in FIVE_CARD_SUBTYPES:
            feats[action, offset + FIVE_CARD_SUBTYPES.index(combo_type)] = 1.0
        offset += len(FIVE_CARD_SUBTYPES)

        # Rank counts: 13.
        for cid in cards:
            feats[action, offset + _rank_index(cid)] += 0.25
        offset += 13

        # Suit counts: 4.
        for cid in cards:
            feats[action, offset + _suit_index(cid)] += 0.25
        offset += 4

        # Scalar summaries.
        feats[action, offset + 0] = float(len(cards)) / 5.0
        feats[action, offset + 1] = (max(cards) / 52.0) if cards else 0.0
        feats[action, offset + 2] = float(sum(cards)) / (52.0 * 5.0) if cards else 0.0
    return feats


@functools.lru_cache(maxsize=1)
def action_features_np():
    return build_action_features()


@functools.lru_cache(maxsize=8)
def action_features_torch(device):
    return torch.tensor(action_features_np(), dtype=torch.float32, device=device)
