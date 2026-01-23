import numpy as np
import torch
import enumerateOptions
import big2Game

from ML_SKHU.features import encode_belief_input, encode_policy_input, belief_targets
from ML_SKHU.mcts import MCTS


def _apply_action(game, action):
    if action == enumerateOptions.passInd:
        game.updateGame(-1)
        return
    opt, n_cards = enumerateOptions.getOptionNC(action)
    game.updateGame(opt, n_cards)


def _uniform_belief(game, perspective_player):
    belief = np.zeros((52, 3), dtype=np.float32)
    targets, mask = belief_targets(game, perspective_player)
    for i in range(52):
        if mask[i] > 0:
            belief[i] = np.array([1.0 / 3.0] * 3, dtype=np.float32)
    return belief


def _min_play_action(game):
    avail = game.returnAvailableActions()
    valid = np.flatnonzero(avail == 1)
    if valid.size == 0:
        return enumerateOptions.passInd
    non_pass = valid[valid != enumerateOptions.passInd]
    if non_pass.size > 0:
        return int(np.min(non_pass))
    return enumerateOptions.passInd


def make_policy_value_fn(belief_model, policy_value_model, device="cpu"):
    def _fn(game, perspective_player):
        if belief_model is None or policy_value_model is None:
            logits = np.zeros((enumerateOptions.passInd + 1,), dtype=np.float32)
            return logits, 0.0
        belief_in = encode_belief_input(game, perspective_player)
        with torch.no_grad():
            b_in = torch.from_numpy(belief_in).float().unsqueeze(0).to(device)
            b_logits = belief_model(b_in)
            b_probs = torch.softmax(b_logits, dim=-1).cpu().numpy()[0]
        policy_in = encode_policy_input(game, perspective_player, b_probs)
        with torch.no_grad():
            p_in = torch.from_numpy(policy_in).float().unsqueeze(0).to(device)
            p_logits, value = policy_value_model(p_in)
        return p_logits.cpu().numpy()[0], float(value.cpu().numpy()[0][0])
    return _fn


def run_selfplay_episode(
    belief_model=None,
    policy_value_model=None,
    n_simulations=50,
    temperature=1.0,
    device="cpu",
    policy_player=1,
    opponent_policy="selfplay",
):
    game = big2Game.big2Game()
    policy_value_fn = make_policy_value_fn(belief_model, policy_value_model, device=device)
    mcts = MCTS(policy_value_fn, n_simulations=n_simulations)

    belief_data = []
    policy_data = []

    while not game.gameOver:
        player = game.playersGo
        belief_in = encode_belief_input(game, player)
        b_target, b_mask = belief_targets(game, player)
        if belief_model is None:
            b_probs = _uniform_belief(game, player)
        else:
            with torch.no_grad():
                b_in = torch.from_numpy(belief_in).float().unsqueeze(0).to(device)
                b_logits = belief_model(b_in)
                b_probs = torch.softmax(b_logits, dim=-1).cpu().numpy()[0]

        policy_in = encode_policy_input(game, player, b_probs)

        belief_data.append((belief_in, b_target, b_mask))

        if opponent_policy == "heuristic" and player != policy_player:
            action = _min_play_action(game)
        else:
            action, visits = mcts.select_action(game, player, temperature=temperature)
            if visits.sum() > 0:
                policy_target = visits / visits.sum()
            else:
                policy_target = np.zeros_like(visits)
                policy_target[action] = 1.0
            if player == policy_player:
                policy_data.append((policy_in, policy_target, player))

        _apply_action(game, action)

    rewards = game.rewards
    policy_data_with_values = []
    for policy_in, policy_target, player in policy_data:
        value_target = 1.0 if rewards[player - 1] > 0 else -1.0
        policy_data_with_values.append((policy_in, policy_target, value_target))

    return belief_data, policy_data_with_values
