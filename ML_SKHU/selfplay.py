import numpy as np
import torch
import enumerateOptions
import big2Game

from ML_SKHU.features import (
    encode_belief_input,
    encode_full_info_with_belief,
    belief_targets,
)
from ML_SKHU.mcts import MCTS


def _apply_action(game, action):
    if action == enumerateOptions.passInd:
        game.updateGame(-1)
        return
    opt, n_cards = enumerateOptions.getOptionNC(action)
    game.updateGame(opt, n_cards)


def _uniform_belief(game, perspective_player):
    belief = np.zeros((52, 3), dtype=np.float32)
    _, mask = belief_targets(game, perspective_player)
    for i in range(52):
        if mask[i] > 0:
            belief[i] = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=np.float32)
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


def _random_action(game):
    avail = game.returnAvailableActions()
    valid = np.flatnonzero(avail == 1)
    if valid.size == 0:
        return enumerateOptions.passInd
    return int(np.random.choice(valid))


def _policy_only_action(model, game, device="cpu"):
    avail = game.returnAvailableActions()
    valid = np.flatnonzero(avail == 1)
    if valid.size == 0:
        return enumerateOptions.passInd
    belief_in = encode_belief_input(game, game.playersGo)
    action_mask = np.isfinite(
        big2Game.convertAvailableActions(avail.astype(np.float32))
    ).astype(np.float32)
    with torch.no_grad():
        p_in = torch.from_numpy(belief_in).float().unsqueeze(0).to(device)
        mask_t = torch.from_numpy(action_mask).float().unsqueeze(0).to(device)
        logits, _ = model(p_in, mask_t)
        action = int(torch.argmax(logits, dim=-1).item())
    return action


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
        policy_in = encode_full_info_with_belief(game, perspective_player, b_probs)
        avail = game.returnAvailableActions().astype(np.float32)
        action_mask = torch.tensor(
            np.isfinite(big2Game.convertAvailableActions(avail)).astype(np.float32)
        ).unsqueeze(0).to(device)
        with torch.no_grad():
            p_in = torch.from_numpy(policy_in).float().unsqueeze(0).to(device)
            p_logits, value = policy_value_model(p_in, action_mask)
        return p_logits.cpu().numpy()[0], float(value.view(-1).cpu().numpy()[0])
    return _fn


def run_selfplay_episode(
    belief_model=None,
    policy_value_model=None,
    policy_only_model=None,
    n_simulations=50,
    temperature=1.0,
    device="cpu",
    policy_player=1,
    opponent_policy="selfplay",
    belief_min_turn=0,
    belief_decay=0.5,
):
    game = big2Game.big2Game()
    use_mcts = n_simulations and n_simulations > 0 and policy_value_model is not None
    if use_mcts:
        policy_value_fn = make_policy_value_fn(belief_model, policy_value_model, device=device)
        mcts = MCTS(policy_value_fn, n_simulations=n_simulations)

    belief_data = []
    policy_data = []
    pending_beliefs = []

    turn_idx = 0
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

        policy_in = encode_full_info_with_belief(game, player, b_probs)

        action_mask = np.isfinite(big2Game.convertAvailableActions(game.returnAvailableActions().astype(np.float32))).astype(np.float32)

        if player != policy_player or not use_mcts:
            if player == policy_player and policy_only_model is not None:
                action = _policy_only_action(policy_only_model, game, device=device)
            elif opponent_policy == "heuristic":
                action = _min_play_action(game)
            else:
                action = _random_action(game)
        else:
            action, visits = mcts.select_action(game, player, temperature=temperature)
            if visits.sum() > 0:
                policy_target = visits / visits.sum()
            else:
                policy_target = np.zeros_like(visits)
                policy_target[action] = 1.0
            policy_data.append((policy_in, policy_target, player, action_mask))

        # When a non-pass happens, backpropagate supervision to prior turns with decay.
        if action != enumerateOptions.passInd and pending_beliefs:
            reveal_turn = turn_idx - 1
            for (p_in, p_target, p_mask, p_turn) in pending_beliefs:
                if p_turn < belief_min_turn:
                    continue
                dist = max(0, reveal_turn - p_turn)
                weight = belief_decay ** dist
                belief_data.append((p_in, p_target, p_mask, float(weight)))
            pending_beliefs.clear()

        # Store current state for potential future supervision.
        if turn_idx >= belief_min_turn:
            pending_beliefs.append((belief_in, b_target, b_mask, turn_idx))

        _apply_action(game, action)
        turn_idx += 1

    rewards = game.rewards
    policy_data_with_values = []
    for policy_in, policy_target, player, action_mask in policy_data:
        value_target = 1.0 if rewards[player - 1] > 0 else -1.0
        policy_data_with_values.append((policy_in, policy_target, value_target, action_mask))

    return belief_data, policy_data_with_values
