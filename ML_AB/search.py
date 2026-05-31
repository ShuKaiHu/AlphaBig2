import time

import numpy as np
import torch

import big2Game
import enumerateOptions

from ML_AB.actions import ACTION_DIM, action_features_torch
from ML_AB.state import action_mask, apply_action, encode_game, public_belief_prior


class Node:
    def __init__(self, game, prior=0.0, parent=None, action=None):
        self.game = game
        self.prior = float(prior)
        self.parent = parent
        self.action = action
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0.0

    @property
    def q(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / float(self.visit_count)


class ModelMCTS:
    def __init__(
        self,
        model,
        device="cpu",
        simulations=32,
        c_puct=1.5,
        value_scale=15.0,
        dirichlet_alpha=0.3,
        dirichlet_frac=0.15,
        max_children=64,
        move_time_limit_sec=0.0,
    ):
        self.model = model
        self.device = device
        self.simulations = int(simulations)
        self.c_puct = float(c_puct)
        self.value_scale = float(value_scale)
        self.dirichlet_alpha = float(dirichlet_alpha)
        self.dirichlet_frac = float(dirichlet_frac)
        self.max_children = int(max_children)
        self.move_time_limit_sec = float(move_time_limit_sec)

    def _terminal_value(self, game, root_player):
        if not game.gameOver:
            return None
        return float(np.tanh(float(game.rewards[root_player - 1]) / self.value_scale))

    def _network(self, game, perspective_player):
        state = encode_game(game, perspective_player, public_belief_prior(game, perspective_player))
        mask = action_mask(game)
        with torch.no_grad():
            card = torch.tensor(state["card_feats"], dtype=torch.float32, device=self.device).unsqueeze(0)
            hist = torch.tensor(state["history_feats"], dtype=torch.float32, device=self.device).unsqueeze(0)
            glob = torch.tensor(state["global_feats"], dtype=torch.float32, device=self.device).unsqueeze(0)
            af = action_features_torch(self.device).unsqueeze(0)
            am = torch.tensor(mask, dtype=torch.float32, device=self.device).unsqueeze(0)
            logits, value, _belief = self.model(card, hist, glob, af, am)
        return logits.cpu().numpy()[0], float(value.cpu().numpy()[0]), mask

    def _expand(self, node, root_player):
        logits, value, mask = self._network(node.game, root_player)
        valid = np.flatnonzero(mask > 0)
        if valid.size == 0:
            return 0.0
        stable = logits[valid] - np.max(logits[valid])
        probs = np.exp(stable)
        probs_sum = probs.sum()
        if probs_sum <= 0:
            probs = np.ones_like(probs) / float(len(probs))
        else:
            probs = probs / probs_sum

        if self.max_children > 0 and valid.size > self.max_children:
            keep_idx = np.argsort(-probs)[: self.max_children]
            valid = valid[keep_idx]
            probs = probs[keep_idx]
            probs = probs / max(float(probs.sum()), 1e-8)

        for action, prior in zip(valid, probs):
            child_game = node.game.clone()
            apply_action(child_game, int(action))
            node.children[int(action)] = Node(child_game, prior=float(prior), parent=node, action=int(action))
        return value

    def _add_noise(self, root):
        if len(root.children) <= 1:
            return
        actions = list(root.children.keys())
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(actions)).astype(np.float32)
        for i, action in enumerate(actions):
            child = root.children[action]
            child.prior = (1.0 - self.dirichlet_frac) * child.prior + self.dirichlet_frac * float(noise[i])

    def _select_child(self, node):
        sqrt_n = np.sqrt(max(1, node.visit_count))
        best_action = None
        best_score = -1e18
        for action, child in node.children.items():
            score = child.q + self.c_puct * child.prior * sqrt_n / (1 + child.visit_count)
            if score > best_score:
                best_score = score
                best_action = action
        return node.children[best_action]

    def search(self, game, root_player, temperature=1.0, add_noise=True):
        root = Node(game.clone())
        self._expand(root, root_player)
        if add_noise:
            self._add_noise(root)

        deadline = None
        if self.move_time_limit_sec > 0:
            deadline = time.perf_counter() + self.move_time_limit_sec

        for _ in range(self.simulations):
            if deadline is not None and time.perf_counter() >= deadline:
                break
            node = root
            path = [node]
            while node.children:
                node = self._select_child(node)
                path.append(node)
            value = self._terminal_value(node.game, root_player)
            if value is None:
                value = self._expand(node, root_player)
            for item in path:
                item.visit_count += 1
                item.value_sum += float(value)

        visits = np.zeros((ACTION_DIM,), dtype=np.float32)
        for action, child in root.children.items():
            visits[action] = float(child.visit_count)
        valid = visits > 0
        if not np.any(valid):
            if root.children:
                priors = np.zeros((ACTION_DIM,), dtype=np.float32)
                for action, child in root.children.items():
                    priors[action] = float(child.prior)
                prior_sum = float(priors.sum())
                if prior_sum > 0:
                    visits = priors / prior_sum
                action = int(max(root.children.items(), key=lambda item: item[1].prior)[0])
                return action, visits
            legal = np.flatnonzero(action_mask(game) > 0)
            action = int(legal[0]) if legal.size else enumerateOptions.passInd
            visits[action] = 1.0
            return action, visits

        if temperature <= 1e-8:
            action = int(np.argmax(visits))
        else:
            probs = np.zeros_like(visits)
            probs[valid] = np.power(visits[valid], 1.0 / max(float(temperature), 1e-8))
            probs = probs / max(float(probs.sum()), 1e-8)
            action = int(np.random.choice(np.arange(ACTION_DIM), p=probs))
        return action, visits


def collect_search_episode(
    model,
    device="cpu",
    simulations=24,
    value_scale=15.0,
    max_turns=240,
    move_time_limit_sec=0.0,
):
    from ML_AB.state import belief_targets

    game = big2Game.big2Game()
    mcts = ModelMCTS(
        model,
        device=device,
        simulations=simulations,
        value_scale=value_scale,
        move_time_limit_sec=move_time_limit_sec,
    )
    samples = []
    turns = 0
    while not game.gameOver and turns < max_turns:
        player = game.playersGo
        b_prior = public_belief_prior(game, player)
        encoded = encode_game(game, player, b_prior)
        mask = action_mask(game)
        b_target, b_mask = belief_targets(game, player)
        temp = 1.0 if turns < 12 else 0.25
        action, visits = mcts.search(game, player, temperature=temp, add_noise=True)
        target = visits.astype(np.float32)
        target = target / max(float(target.sum()), 1e-8)
        samples.append(
            {
                "card_feats": encoded["card_feats"],
                "history_feats": encoded["history_feats"],
                "global_feats": encoded["global_feats"],
                "action_mask": mask,
                "action": int(action),
                "policy_target": target,
                "player": player,
                "belief_target": b_target,
                "belief_mask": b_mask,
            }
        )
        apply_action(game, action)
        turns += 1

    if game.gameOver:
        rewards = np.array(game.rewards, dtype=np.float32)
    else:
        counts = np.array([len(game.currentHands[p]) for p in range(1, 5)], dtype=np.float32)
        leader = int(np.argmin(counts))
        rewards = -counts
        rewards[leader] = float(np.sum(counts) - counts[leader])
    for sample in samples:
        sample["value_target"] = np.tanh(float(rewards[sample["player"] - 1]) / float(value_scale))
    return samples, rewards
