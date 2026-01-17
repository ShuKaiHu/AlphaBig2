import math
import random

import numpy as np

import big2Game
import enumerateOptions
import gameLogic
import torch


def list_valid_actions(game):
    if game.gameOver or game.currentHands[game.playersGo].size == 0:
        return []
    available = game.returnAvailableActions()
    return [int(i) for i in np.flatnonzero(available == 1)]


def decode_action(game, action):
    opt, nC = enumerateOptions.getOptionNC(action)
    if nC == 1:
        hand = np.array([game.currentHands[game.playersGo][opt]])
    elif nC == 2:
        hand = game.currentHands[game.playersGo][enumerateOptions.inverseTwoCardIndices[opt]]
    else:
        hand = game.currentHands[game.playersGo][enumerateOptions.inverseFiveCardIndices[opt]]
    return opt, nC, hand


def choose_heuristic_action(game):
    valid = list_valid_actions(game)
    if not valid:
        return enumerateOptions.passInd
    non_pass = [a for a in valid if a != enumerateOptions.passInd]
    if not non_pass:
        return enumerateOptions.passInd

    if game.control == 1:
        for nC in (5, 2, 1):
            candidates = []
            for action in non_pass:
                _opt, a_nC = enumerateOptions.getOptionNC(action)
                if a_nC == nC:
                    candidates.append(action)
            if candidates:
                return min(candidates)
        return min(non_pass)

    prev_hand = game.handsPlayed[game.goIndex - 1].hand
    prev_size = len(prev_hand)
    candidates = []
    for action in non_pass:
        _opt, a_nC = enumerateOptions.getOptionNC(action)
        if a_nC == prev_size:
            candidates.append(action)
    if candidates:
        return min(candidates)

    override_candidates = []
    for action in non_pass:
        _opt, a_nC, hand = decode_action(game, action)
        if a_nC == 5 and (gameLogic.isFourOfAKind(hand) or gameLogic.isStraightFlush(hand)):
            override_candidates.append(action)
    if override_candidates:
        return min(override_candidates)

    return enumerateOptions.passInd


def _belief_input_from_game(model, game):
    input_dim = model.net[0].in_features
    own_hand = np.zeros((52,), dtype=np.float32)
    for card in game.currentHands[1]:
        own_hand[card - 1] = 1.0

    played = np.zeros((52,), dtype=np.float32)
    played_mask = np.sum(game.cardsPlayed, axis=0)
    played[played_mask > 0] = 1.0

    counts = np.array(
        [
            game.currentHands[1].size,
            game.currentHands[2].size,
            game.currentHands[3].size,
            game.currentHands[4].size,
        ],
        dtype=np.float32,
    )
    counts = counts / 13.0

    tail = np.concatenate([own_hand, played, counts], axis=0)
    history_len = input_dim - tail.size
    if history_len < 0:
        raise ValueError("belief model input too small for tail features")
    history = np.zeros((history_len,), dtype=np.float32)
    return np.concatenate([history, tail], axis=0)


def _belief_from_model(model, game):
    x = _belief_input_from_game(model, game)
    obs_tensor = torch.from_numpy(x).unsqueeze(0)
    with torch.no_grad():
        logits = model(obs_tensor)
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
    return probs


def sample_opponent_hands(game, belief_probs=None, temperature=1.0):
    # Determinization: sample unknown cards uniformly.
    known = set(game.currentHands[1].tolist())
    played = np.flatnonzero(np.sum(game.cardsPlayed, axis=0) > 0) + 1
    known.update(played.tolist())
    unknown = [c for c in range(1, 53) if c not in known]

    sizes = {
        2: int(game.currentHands[2].size),
        3: int(game.currentHands[3].size),
        4: int(game.currentHands[4].size),
    }
    hands = {2: [], 3: [], 4: []}

    if belief_probs is None:
        random.shuffle(unknown)
        idx = 0
        for pid in (2, 3, 4):
            take = sizes[pid]
            hands[pid] = np.array(unknown[idx : idx + take], dtype=int)
            hands[pid].sort()
            idx += take
        return hands

    # belief_probs shape: [52, 3] for P2/P3/P4
    remaining = {2: sizes[2], 3: sizes[3], 4: sizes[4]}
    for card in unknown:
        probs = belief_probs[card - 1].copy()
        if temperature != 1.0:
            probs = np.power(probs, 1.0 / temperature)
        # zero out players with no remaining slots
        for idx, pid in enumerate([2, 3, 4]):
            if remaining[pid] == 0:
                probs[idx] = 0.0
        if probs.sum() == 0:
            # fallback to any player with remaining slots
            candidates = [pid for pid in (2, 3, 4) if remaining[pid] > 0]
            pid = random.choice(candidates)
        else:
            probs = probs / probs.sum()
            pid = int(np.random.choice([2, 3, 4], p=probs))
        hands[pid].append(card)
        remaining[pid] -= 1

    for pid in (2, 3, 4):
        hands[pid] = np.array(hands[pid], dtype=int)
    for pid in (2, 3, 4):
        hands[pid].sort()
    return hands


def apply_determinization(game, sampled_hands):
    for pid, hand in sampled_hands.items():
        game.currentHands[pid] = hand.copy()


class MCTSNode:
    def __init__(self, game, root_player, parent=None, action=None, priors=None):
        self.game = game
        self.root_player = root_player
        self.parent = parent
        self.action = action
        self.children = {}
        self.untried_actions = list_valid_actions(game)
        self.N = 0
        self.W = 0.0
        self.priors = priors or {}

    @property
    def Q(self):
        return self.W / self.N if self.N > 0 else 0.0

    def is_terminal(self):
        return self.game.gameOver == 1

    def expand(self, policy_model=None):
        if self.priors:
            action = max(self.untried_actions, key=lambda a: self.priors.get(a, 0.0))
            self.untried_actions.remove(action)
        else:
            action = self.untried_actions.pop()
        child_game = self.game.clone()
        if action == enumerateOptions.passInd:
            child_game.updateGame(-1)
        else:
            opt, nC, _hand = decode_action(child_game, action)
            child_game.updateGame(opt, nC)
        priors = policy_priors(child_game, policy_model)
        child = MCTSNode(child_game, self.root_player, parent=self, action=action, priors=priors)
        self.children[action] = child
        return child

    def best_child(self, c):
        best_score = -1e9
        best = None
        child_count = max(len(self.children), 1)
        for action, child in self.children.items():
            prior = self.priors.get(action, 1.0 / child_count)
            ucb = child.Q + c * prior * math.sqrt(self.N + 1.0) / (1.0 + child.N)
            if ucb > best_score:
                best_score = ucb
                best = child
        return best

    def backup(self, reward):
        self.N += 1
        self.W += reward
        if self.parent is not None:
            self.parent.backup(reward)


def rollout(game, root_player, max_steps=200, risk_aversion=0.0, rollout_policy=None):
    game = game.clone()
    steps = 0
    while not game.gameOver and steps < max_steps:
        if rollout_policy is None:
            action = choose_heuristic_action(game)
        else:
            obs = np.asarray(game.neuralNetworkInputs[game.playersGo], dtype=np.float32)
            avail = game.returnAvailableActions().astype(np.float32)
            action_mask = np.isfinite(big2Game.convertAvailableActions(avail))
            with torch.no_grad():
                obs_t = torch.from_numpy(obs).unsqueeze(0)
                mask_t = torch.from_numpy(action_mask.astype(np.float32)).unsqueeze(0)
                logits = rollout_policy(obs_t, mask_t).squeeze(0).cpu().numpy()
            action = int(np.argmax(logits))
        if action == enumerateOptions.passInd:
            game.updateGame(-1)
        else:
            opt, nC, _hand = decode_action(game, action)
            game.updateGame(opt, nC)
        steps += 1
    if game.gameOver:
        reward = float(game.rewards[root_player - 1])
        if reward < 0 and risk_aversion > 0:
            reward *= 1.0 + risk_aversion
        return reward
    return 0.0


def policy_priors(game, policy_model):
    if policy_model is None:
        return {}
    obs = np.asarray(game.neuralNetworkInputs[game.playersGo], dtype=np.float32)
    avail = game.returnAvailableActions().astype(np.float32)
    valid = np.flatnonzero(avail == 1)
    if valid.size == 0:
        return {}
    action_mask = np.isfinite(big2Game.convertAvailableActions(avail))
    with torch.no_grad():
        obs_t = torch.from_numpy(obs).unsqueeze(0)
        mask_t = torch.from_numpy(action_mask.astype(np.float32)).unsqueeze(0)
        logits = policy_model(obs_t, mask_t).squeeze(0).cpu().numpy()
    logits_valid = logits[valid]
    max_logit = np.max(logits_valid)
    exp = np.exp(logits_valid - max_logit)
    total = exp.sum()
    if total <= 0:
        return {}
    probs = exp / total
    return {int(a): float(p) for a, p in zip(valid, probs)}


def mcts_policy(
    game,
    simulations=200,
    c=1.4,
    belief_model=None,
    temperature=1.0,
    risk_aversion=0.0,
    max_rollout_steps=200,
    rollout_policy=None,
    policy_model=None,
):
    root_player = game.playersGo
    root_game = game.clone()
    belief_probs = None
    if belief_model is not None:
        belief_probs = _belief_from_model(belief_model, root_game)
    sampled = sample_opponent_hands(root_game, belief_probs, temperature=temperature)
    apply_determinization(root_game, sampled)
    root = MCTSNode(root_game, root_player, priors=policy_priors(root_game, policy_model))

    for _ in range(simulations):
        node = root
        while node.untried_actions == [] and node.children:
            node = node.best_child(c)
        if node.untried_actions:
            node = node.expand(policy_model=policy_model)
        reward = rollout(
            node.game,
            root_player,
            max_steps=max_rollout_steps,
            risk_aversion=risk_aversion,
            rollout_policy=rollout_policy,
        )
        node.backup(reward)

    valid = list_valid_actions(game)
    if not valid:
        return np.zeros((enumerateOptions.nActions[5] + 1,), dtype=np.float32)
    counts = np.zeros((enumerateOptions.nActions[5] + 1,), dtype=np.float32)
    for action, child in root.children.items():
        counts[action] = float(child.N)
    if counts.sum() == 0:
        for a in valid:
            counts[a] = 1.0
    probs = counts / counts.sum()
    return probs


def mcts_search(
    game,
    simulations=200,
    c=1.4,
    max_rollout_steps=200,
    belief_model=None,
    temperature=1.0,
    risk_aversion=0.0,
    rollout_policy=None,
    policy_model=None,
):
    probs = mcts_policy(
        game,
        simulations=simulations,
        c=c,
        belief_model=belief_model,
        temperature=temperature,
        risk_aversion=risk_aversion,
        max_rollout_steps=max_rollout_steps,
        rollout_policy=rollout_policy,
        policy_model=policy_model,
    )
    valid = list_valid_actions(game)
    if not valid:
        return enumerateOptions.passInd
    masked = np.zeros_like(probs)
    for a in valid:
        masked[a] = probs[a]
    if masked.sum() == 0:
        return enumerateOptions.passInd
    return int(np.argmax(masked))
