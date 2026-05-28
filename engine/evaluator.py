import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import enumerateOptions
from engine.env import Big2Env, PASS_IDX
from engine.features import encode_static, encode_history_steps


def _model_action(model, env: Big2Env) -> int:
    """Greedy policy action (no MCTS)."""
    game = env.game
    player = env.current_player
    static = encode_static(game, player)
    history = encode_history_steps(game)
    valid = env.get_valid_actions()
    probs, _ = model.predict(static, history, valid)
    masked = probs * valid
    if masked.sum() == 0:
        return PASS_IDX
    return int(np.argmax(masked))


def _heuristic_action(env: Big2Env) -> int:
    """Smarter heuristic — same logic as self_play._heuristic_action."""
    import enumerateOptions as _eo
    valid_mask = env.get_valid_actions()
    valid = np.flatnonzero(valid_mask)
    non_pass = valid[valid != PASS_IDX]
    if len(non_pass) == 0:
        return PASS_IDX
    game = env.game
    if game.control == 1:
        five_card = [a for a in non_pass if a >= _eo.FIVE_OFFSET]
        pairs      = [a for a in non_pass if _eo.PAIR_OFFSET <= a < _eo.FIVE_OFFSET]
        singles    = [a for a in non_pass if a < _eo.PAIR_OFFSET]
        if five_card:
            return int(np.random.choice(five_card))
        if pairs:
            return int(pairs[0])
        return int(singles[0])
    else:
        return int(non_pass[0])


def evaluate_vs_heuristic(model, n_games: int = 200, learner_player: int = 1):
    """
    Evaluate model (as `learner_player`) against 3 heuristic opponents.
    Returns (win_rate, avg_score).
    """
    wins = 0
    scores = []
    env = Big2Env()

    for _ in range(n_games):
        env.reset()
        while not env.done:
            player = env.current_player
            if player == learner_player:
                action = _model_action(model, env)
            else:
                action = _heuristic_action(env)
            rewards, done = env.step(action)
            if done:
                score = float(rewards[learner_player - 1])
                scores.append(score)
                wins += int(score > 0)
                break

    return wins / n_games, float(np.mean(scores)) if scores else 0.0


def evaluate_model_vs_model(model_a, model_b, n_games: int = 200):
    """
    model_a plays as P1, model_b plays as P2–P4.
    Returns (win_rate_a, avg_score_a).
    """
    wins = 0
    scores = []
    env = Big2Env()

    for _ in range(n_games):
        env.reset()
        while not env.done:
            player = env.current_player
            action = _model_action(model_a if player == 1 else model_b, env)
            rewards, done = env.step(action)
            if done:
                score = float(rewards[0])
                scores.append(score)
                wins += int(score > 0)
                break

    return wins / n_games, float(np.mean(scores)) if scores else 0.0
