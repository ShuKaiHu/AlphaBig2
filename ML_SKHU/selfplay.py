import numpy as np
import torch

import big2Game
import enumerateOptions

from ML_SKHU.features import (
    encode_belief_input,
    encode_policy_input,
    belief_targets,
)
from ML_SKHU.mcts import MCTS


def _apply_action(game, action):
    if action == enumerateOptions.passInd:
        game.updateGame(-1)
    else:
        opt, n = enumerateOptions.getOptionNC(action)
        game.updateGame(opt, n)


def _random_action(game):
    avail = game.returnAvailableActions()
    valid = np.flatnonzero(avail == 1)
    if valid.size == 0:
        return enumerateOptions.passInd
    return int(np.random.choice(valid))


def _heuristic_action(game):
    avail = game.returnAvailableActions()
    valid = np.flatnonzero(avail == 1)
    if valid.size == 0:
        return enumerateOptions.passInd
    non_pass = valid[valid != enumerateOptions.passInd]
    if non_pass.size > 0:
        return int(np.min(non_pass))
    return int(enumerateOptions.passInd)


def _oracle_belief_probs(game):
    probs = np.zeros((52, 3), dtype=np.float32)
    for abs_player, ch in [(2, 0), (3, 1), (4, 2)]:
        for cid in game.currentHands[abs_player]:
            probs[int(cid) - 1, ch] = 1.0
    return probs


def _belief_probs(game, player, belief_model, device, belief_mode="model"):
    if belief_mode == "oracle":
        return _oracle_belief_probs(game)
    if belief_mode == "uniform":
        return np.full((52, 3), 1.0 / 3.0, dtype=np.float32)
    if belief_model is None:
        # uniform for P2/P3/P4
        return np.full((52, 3), 1.0 / 3.0, dtype=np.float32)
    x = encode_belief_input(game, player)
    with torch.no_grad():
        t = torch.from_numpy(x).float().unsqueeze(0).to(device)
        logits = belief_model(t)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    return probs.astype(np.float32)


def make_policy_fn(belief_model, policy_model, device="cpu", belief_mode="model"):
    def _fn(game, perspective_player):
        b_probs = _belief_probs(
            game, perspective_player, belief_model, device=device, belief_mode=belief_mode
        )
        p_in = encode_policy_input(game, perspective_player, b_probs)
        avail = game.returnAvailableActions().astype(np.float32)
        action_mask = np.isfinite(big2Game.convertAvailableActions(avail.copy())).astype(np.float32)
        with torch.no_grad():
            x = torch.from_numpy(p_in).float().unsqueeze(0).to(device)
            m = torch.from_numpy(action_mask).float().unsqueeze(0).to(device)
            logits = policy_model(x, m).cpu().numpy()[0]
        return logits
    return _fn


def make_value_fn(belief_model, value_model, device="cpu", belief_mode="model"):
    def _fn(game, perspective_player):
        b_probs = _belief_probs(
            game, perspective_player, belief_model, device=device, belief_mode=belief_mode
        )
        p_in = encode_policy_input(game, perspective_player, b_probs)
        with torch.no_grad():
            x = torch.from_numpy(p_in).float().unsqueeze(0).to(device)
            v = value_model(x).view(-1).cpu().numpy()[0]
        return float(v)
    return _fn


def run_selfplay_episode(
    belief_model,
    policy_model,
    value_model,
    n_simulations=50,
    temperature=1.0,
    device="cpu",
    policy_player=1,
    opponent_policy="heuristic",
    value_scale=15.0,
    belief_mode="model",
    target_temperature=1.0,
    dirichlet_alpha=0.3,
    dirichlet_frac=0.25,
    add_root_noise=True,
):
    game = big2Game.big2Game()
    policy_fn = make_policy_fn(
        belief_model, policy_model, device=device, belief_mode=belief_mode
    )
    value_fn = make_value_fn(
        belief_model, value_model, device=device, belief_mode=belief_mode
    )
    mcts = MCTS(
        policy_fn=policy_fn,
        value_fn=value_fn,
        n_simulations=n_simulations,
        dirichlet_alpha=dirichlet_alpha,
        dirichlet_frac=dirichlet_frac,
    )

    belief_data = []
    policy_data = []

    while not game.gameOver:
        player = game.playersGo

        # collect belief supervision from public states
        b_in = encode_belief_input(game, player)
        b_target, b_mask = belief_targets(game, player)
        belief_data.append((b_in, b_target, b_mask, 1.0))

        if player == policy_player:
            b_probs = _belief_probs(
                game, player, belief_model, device=device, belief_mode=belief_mode
            )
            p_in = encode_policy_input(game, player, b_probs)
            avail = game.returnAvailableActions().astype(np.float32)
            action_mask = np.isfinite(big2Game.convertAvailableActions(avail.copy())).astype(np.float32)

            if n_simulations > 0:
                action, visits = mcts.select_action(
                    game,
                    player,
                    temperature=temperature,
                    value_scale=value_scale,
                    add_root_noise=add_root_noise,
                )
                target = np.zeros_like(visits, dtype=np.float32)
                s = visits.sum()
                if s > 0:
                    t = max(float(target_temperature), 1e-8)
                    if t <= 1e-8:
                        target[np.argmax(visits)] = 1.0
                    else:
                        shaped = np.power(visits, 1.0 / t)
                        ss = shaped.sum()
                        if ss > 0:
                            target = shaped / ss
                        else:
                            target = visits / s
                else:
                    target[action] = 1.0
            else:
                logits = policy_fn(game, player)
                valid = np.flatnonzero(action_mask > 0)
                if valid.size == 0:
                    action = enumerateOptions.passInd
                else:
                    action = int(valid[np.argmax(logits[valid])])
                target = np.zeros((enumerateOptions.passInd + 1,), dtype=np.float32)
                target[action] = 1.0

            policy_data.append((p_in, target, action_mask))
        else:
            if opponent_policy == "heuristic":
                action = _heuristic_action(game)
            else:
                action = _random_action(game)

        _apply_action(game, action)

    reward = float(game.rewards[policy_player - 1])
    value_target = np.tanh(reward / float(value_scale))
    policy_data_with_values = []
    for p_in, target, action_mask in policy_data:
        policy_data_with_values.append((p_in, target, value_target, action_mask))

    return belief_data, policy_data_with_values
