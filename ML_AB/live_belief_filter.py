import math
from collections import defaultdict

import numpy as np

import big2Game
import enumerateOptions
import gameLogic

from ML_AB.actions import action_cards, action_combo_type
from ML_AB.state import action_mask


def _cards(entry):
    hand = entry.get("hand")
    if hand is None:
        return []
    return sorted(int(card) for card in hand)


def _played_by_player(history):
    played = defaultdict(list)
    for entry in history:
        try:
            player = int(entry.get("player"))
        except (TypeError, ValueError):
            continue
        if entry.get("pass", False):
            continue
        played[player].extend(_cards(entry))
    return played


def _rank_value(card_id):
    return int(math.ceil(int(card_id) / 4.0))


def _strength_key(action):
    cards = np.array(action_cards(int(action)), dtype=int)
    if cards.size == 0:
        return (0,)

    combo_type = action_combo_type(int(action))
    if combo_type in {"single", "pair"}:
        return (int(np.max(cards)),)
    if combo_type in {"straight", "straight_flush"}:
        rank = gameLogic.straightRank(cards.copy())
        if rank is None:
            return (-1, -1)
        return (int(rank[0]), int(rank[1]))
    if combo_type == "full_house":
        full_house = gameLogic.isFullHouse(cards.copy())
        return (int(full_house[1]) if full_house[0] else -1, int(np.max(cards)))
    if combo_type == "four_of_a_kind":
        quad_value = gameLogic.fourOfAKindValue(cards.copy())
        return (int(quad_value) if quad_value is not None else -1, int(np.max(cards)))
    return (int(cards.size), int(np.max(cards)))


def _same_family(a_action, b_action):
    return action_combo_type(int(a_action)) == action_combo_type(int(b_action))


def _build_replay_game(public_game, assignment, history):
    played = _played_by_player(history)
    game = big2Game.big2Game()
    game.currentHands = {}
    for player in range(1, 5):
        if player == 1:
            remaining = [int(card) for card in public_game.currentHands[1] if int(card) > 0]
        else:
            remaining = [int(card) for card in assignment.get(player, []) if int(card) > 0]
        game.currentHands[player] = np.sort(
            np.array(remaining + [int(card) for card in played[player]], dtype=int)
        )

    game.cardsPlayed = np.zeros((4, 52), dtype=int)
    first_player = None
    first_play = None
    for entry in history:
        try:
            first_player = int(entry.get("player"))
        except (TypeError, ValueError):
            continue
        if not entry.get("pass", False):
            first_play = _cards(entry)
        break
    game.playersGo = int(first_player or getattr(public_game, "playersGo", 1))
    game.passCount = 0
    game.passedThisRound = {1: False, 2: False, 3: False, 4: False}
    game.lastPlayedPlayer = game.playersGo
    game.control = 1
    game.goIndex = 1
    game.handsPlayed = {}
    game.actionHistory = []
    game.gameOver = 0
    game.rewards = np.zeros((4,))
    game.goCounter = 0

    club3_holder = None
    for player in range(1, 5):
        if 1 in set(int(card) for card in game.currentHands[player]):
            club3_holder = player
            break
    game.club3Player = int(club3_holder or game.playersGo)
    game.mustPlayClub3 = bool(first_play and 1 in first_play)
    return game


def _play_likelihood(game, action):
    valid = np.flatnonzero(action_mask(game) > 0)
    play_actions = [int(candidate) for candidate in valid if int(candidate) != enumerateOptions.passInd]
    if int(action) not in play_actions:
        return 1.0e-8, {
            "illegal_observed_plays": 1,
            "non_minimal_response_plays": 0,
            "control_single_with_combo_available": 0,
        }

    diagnostics = {
        "illegal_observed_plays": 0,
        "non_minimal_response_plays": 0,
        "control_single_with_combo_available": 0,
    }
    probability = 0.95
    observed_cards = action_cards(int(action))

    if not bool(game.control):
        observed_key = _strength_key(action)
        lower_same_family = [
            candidate
            for candidate in play_actions
            if _same_family(candidate, action) and _strength_key(candidate) < observed_key
        ]
        if lower_same_family:
            diagnostics["non_minimal_response_plays"] = 1
            probability *= max(0.08, math.exp(-0.35 * float(len(lower_same_family))))
    else:
        if len(observed_cards) == 1:
            has_five = any(len(action_cards(candidate)) == 5 for candidate in play_actions)
            has_pair = any(len(action_cards(candidate)) == 2 for candidate in play_actions)
            if has_five:
                probability *= 0.35
                diagnostics["control_single_with_combo_available"] = 1
            elif has_pair:
                probability *= 0.65
                diagnostics["control_single_with_combo_available"] = 1

    return float(max(min(probability, 0.99), 1.0e-6)), diagnostics


def _pass_likelihood(game):
    diagnostics = {
        "passes_with_legal_beat": 0,
        "passes_without_legal_beat": 0,
        "passes_under_control": 0,
    }
    valid = np.flatnonzero(action_mask(game) > 0)
    play_actions = [int(action) for action in valid if int(action) != enumerateOptions.passInd]
    if bool(game.control):
        diagnostics["passes_under_control"] = 1
        return 0.02, diagnostics
    if play_actions:
        diagnostics["passes_with_legal_beat"] = 1
        return 0.12, diagnostics
    diagnostics["passes_without_legal_beat"] = 1
    return 0.98, diagnostics


def _safe_update(game, entry):
    if entry.get("pass", False):
        game.updateGame(-1)
        return
    cards = _cards(entry)
    game.updateGame(cards, len(cards))


def history_log_likelihood(public_game, assignment):
    """Score a hidden-card assignment by how well it explains public actions."""
    history = list(getattr(public_game, "actionHistory", []) or [])
    diagnostics = {
        "history_events": int(len(history)),
        "turn_mismatches": 0,
        "illegal_observed_plays": 0,
        "non_minimal_response_plays": 0,
        "control_single_with_combo_available": 0,
        "passes_with_legal_beat": 0,
        "passes_without_legal_beat": 0,
        "passes_under_control": 0,
        "replay_errors": 0,
        "final_hand_mismatches": 0,
        "inferred_turn_jumps": 0,
    }
    if not history:
        return 0.0, diagnostics

    game = _build_replay_game(public_game, assignment, history)
    logp = 0.0
    for entry in history:
        try:
            actor = int(entry.get("player"))
        except (TypeError, ValueError):
            continue
        if actor not in (1, 2, 3, 4):
            continue
        if int(game.playersGo) != actor:
            diagnostics["turn_mismatches"] += 1
            diagnostics["inferred_turn_jumps"] += 1
            game.playersGo = actor
            if not entry.get("pass", False):
                # The live tracker may miss the three intervening passes that
                # return control to the same player. Treat non-pass turn jumps
                # as an inferred control break instead of declaring the
                # observed play impossible for this hidden-card sample.
                game.control = 1
                game.passCount = 0
                game.passedThisRound = {1: False, 2: False, 3: False, 4: False}
                game.lastPlayedPlayer = actor
                logp += math.log(0.95)
            else:
                logp += math.log(0.8)

        if entry.get("pass", False):
            probability, local = _pass_likelihood(game)
        else:
            try:
                action = enumerateOptions.action_index_from_cards(_cards(entry))
                probability, local = _play_likelihood(game, action)
            except Exception:
                probability = 1.0e-8
                local = {"illegal_observed_plays": 1}

        for key, value in local.items():
            diagnostics[key] = diagnostics.get(key, 0) + int(value)
        logp += math.log(max(float(probability), 1.0e-12))
        try:
            _safe_update(game, entry)
        except Exception:
            diagnostics["replay_errors"] += 1
            logp += math.log(1.0e-8)
            if not entry.get("pass", False):
                for card in _cards(entry):
                    if 1 <= int(card) <= 52:
                        game.cardsPlayed[actor - 1][int(card) - 1] = 1

    expected = {1: np.sort(np.asarray(public_game.currentHands[1], dtype=int))}
    for player in (2, 3, 4):
        expected[player] = np.sort(np.asarray(assignment.get(player, []), dtype=int))
    if not bool(getattr(game, "gameOver", 0)):
        for player in range(1, 5):
            actual = np.sort(np.asarray(game.currentHands[player], dtype=int))
            if actual.shape != expected[player].shape or not np.array_equal(actual, expected[player]):
                diagnostics["final_hand_mismatches"] += 1
                logp += math.log(0.15)

    return float(logp), diagnostics


def summarize_posterior_assignments(assignments, weights=None):
    if not assignments:
        return {}
    if weights is None:
        weights = np.ones((len(assignments),), dtype=np.float64) / float(len(assignments))
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / max(float(weights.sum()), 1.0e-12)

    def owner_has(player, predicate):
        total = 0.0
        for weight, assignment in zip(weights, assignments):
            cards = [int(card) for card in assignment.get(player, [])]
            if any(predicate(card) for card in cards):
                total += float(weight)
        return float(total)

    return {
        "next_has_any_two_prob": owner_has(2, lambda card: _rank_value(card) == 13),
        "prev_has_any_ace_prob": owner_has(4, lambda card: _rank_value(card) == 12),
        "top_has_any_two_prob": owner_has(3, lambda card: _rank_value(card) == 13),
    }
