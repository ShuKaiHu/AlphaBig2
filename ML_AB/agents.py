import numpy as np
import torch

import enumerateOptions

from ML_AB.actions import ACTION_DIM, action_cards, action_combo_type, action_features_torch
from ML_AB.state import action_mask, apply_action, encode_game, public_belief_prior


def random_action(game):
    valid = np.flatnonzero(action_mask(game) > 0)
    if valid.size == 0:
        return enumerateOptions.passInd
    return int(np.random.choice(valid))


def heuristic_action(game):
    valid = np.flatnonzero(action_mask(game) > 0)
    if valid.size == 0:
        return enumerateOptions.passInd
    non_pass = valid[valid != enumerateOptions.passInd]
    if non_pass.size > 0:
        return int(np.min(non_pass))
    return enumerateOptions.passInd


class ModelAgent:
    def __init__(self, model, device="cpu", temperature=0.0):
        self.model = model
        self.device = device
        self.temperature = float(temperature)

    def action_logits(self, game, perspective_player):
        state = encode_game(game, perspective_player, public_belief_prior(game, perspective_player))
        mask = action_mask(game)
        with torch.no_grad():
            card = torch.tensor(state["card_feats"], dtype=torch.float32, device=self.device).unsqueeze(0)
            hist = torch.tensor(state["history_feats"], dtype=torch.float32, device=self.device).unsqueeze(0)
            glob = torch.tensor(state["global_feats"], dtype=torch.float32, device=self.device).unsqueeze(0)
            af = action_features_torch(self.device).unsqueeze(0)
            am = torch.tensor(mask, dtype=torch.float32, device=self.device).unsqueeze(0)
            logits, value, _belief = self.model(card, hist, glob, af, am)
        return logits.cpu().numpy()[0], float(value.cpu().numpy()[0])

    def select_action(self, game, perspective_player):
        logits, _value = self.action_logits(game, perspective_player)
        valid = np.flatnonzero(action_mask(game) > 0)
        if valid.size == 0:
            return enumerateOptions.passInd
        if self.temperature <= 1e-8:
            return int(valid[np.argmax(logits[valid])])
        stable = logits[valid] - np.max(logits[valid])
        probs = np.exp(stable / self.temperature)
        probs /= probs.sum()
        return int(np.random.choice(valid, p=probs))


class RerankAgent(ModelAgent):
    """Inference-time policy reranker focused on reducing terminal card penalties.

    This keeps the checkpoint unchanged and only adjusts legal-action scores. It
    is intentionally conservative: bonuses mainly apply when the player has
    control, where choosing a five-card hand instead of a single is a real
    strategic choice rather than a forced response.
    """

    DEFAULT_SUBTYPE_BONUS = {
        "straight": 0.9,
        "flush": 1.0,
        "full_house": 1.4,
        "four_of_a_kind": 1.8,
        "straight_flush": 2.2,
    }

    def __init__(
        self,
        model,
        device="cpu",
        temperature=0.0,
        control_five_bonus=1.2,
        card_count_bonus=0.12,
        finish_bonus=3.0,
        urgent_opponent_count=3,
        urgent_five_bonus=1.0,
        preserve_five_card_penalty=0.25,
        subtype_bonus=None,
    ):
        super().__init__(model, device=device, temperature=temperature)
        self.control_five_bonus = float(control_five_bonus)
        self.card_count_bonus = float(card_count_bonus)
        self.finish_bonus = float(finish_bonus)
        self.urgent_opponent_count = int(urgent_opponent_count)
        self.urgent_five_bonus = float(urgent_five_bonus)
        self.preserve_five_card_penalty = float(preserve_five_card_penalty)
        self.subtype_bonus = dict(self.DEFAULT_SUBTYPE_BONUS)
        if subtype_bonus:
            self.subtype_bonus.update(subtype_bonus)

    def reranked_logits(self, game, perspective_player):
        logits, value = super().action_logits(game, perspective_player)
        scores = logits.copy()
        valid = np.flatnonzero(action_mask(game) > 0)
        if valid.size == 0:
            return scores, value

        hand_count = len(game.currentHands[perspective_player])
        opponent_counts = [
            len(game.currentHands[p])
            for p in range(1, 5)
            if p != int(perspective_player)
        ]
        urgent = bool(opponent_counts and min(opponent_counts) <= self.urgent_opponent_count)
        five_card_pressure = _five_card_pressure(valid) if game.control else {}

        for action in valid:
            cards = action_cards(int(action))
            n_cards = len(cards)
            if n_cards == 0:
                continue
            if self.card_count_bonus:
                scores[action] += self.card_count_bonus * float(n_cards - 1)
            if n_cards == hand_count:
                scores[action] += self.finish_bonus
            combo_type = action_combo_type(int(action))
            if n_cards == 5:
                scores[action] += self.subtype_bonus.get(combo_type, 0.0)
                if game.control:
                    scores[action] += self.control_five_bonus
                if urgent:
                    scores[action] += self.urgent_five_bonus
            elif five_card_pressure and self.preserve_five_card_penalty > 0:
                pressure = sum(five_card_pressure.get(int(card), 0.0) for card in cards)
                scores[action] -= self.preserve_five_card_penalty * pressure
        return scores, value

    def action_logits(self, game, perspective_player):
        return self.reranked_logits(game, perspective_player)


def _five_card_pressure(valid_actions):
    five_actions = [int(a) for a in valid_actions if len(action_cards(int(a))) == 5]
    if not five_actions:
        return {}
    counts = {}
    for action in five_actions:
        combo_type = action_combo_type(action)
        weight = RerankAgent.DEFAULT_SUBTYPE_BONUS.get(combo_type, 0.5)
        for card in action_cards(action):
            counts[int(card)] = counts.get(int(card), 0.0) + weight
    scale = max(counts.values()) if counts else 1.0
    return {card: value / scale for card, value in counts.items()}


def play_game(policy_fn_by_player):
    import big2Game

    game = big2Game.big2Game()
    while not game.gameOver:
        p = game.playersGo
        action = policy_fn_by_player[p](game, p)
        apply_action(game, action)
    return game.rewards.copy()
