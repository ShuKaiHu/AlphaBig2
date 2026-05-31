import numpy as np

import big2Game

from ML_AB.agents import heuristic_action, random_action
from ML_AB.state import action_mask, apply_action, belief_targets, encode_game, public_belief_prior


def _policy_name_action(name, model_agent=None):
    if name == "heuristic":
        return lambda game, player: heuristic_action(game)
    if name == "random":
        return lambda game, player: random_action(game)
    if name == "model":
        if model_agent is None:
            raise ValueError("model policy requested without model_agent")
        return lambda game, player: model_agent.select_action(game, player)
    raise ValueError(f"unknown policy: {name}")


def _unfinished_rewards(game):
    rewards = np.zeros((4,), dtype=np.float32)
    counts = np.array([len(game.currentHands[p]) for p in range(1, 5)], dtype=np.float32)
    leader = int(np.argmin(counts))
    for i in range(4):
        rewards[i] = -counts[i]
    rewards[leader] = float(np.sum(counts) - counts[leader])
    return rewards


def collect_episode(policy_mix=("heuristic",), model_agent=None, value_scale=15.0, max_turns=300):
    """Collect public-state samples from one game.

    The acting policy may be heuristic/random/model. Targets always use the
    actual action taken, and value targets use final rewards from each state's
    perspective player.
    """
    game = big2Game.big2Game()
    samples = []
    player_policy = {}
    for p in range(1, 5):
        name = np.random.choice(policy_mix)
        player_policy[p] = _policy_name_action(str(name), model_agent=model_agent)

    turns = 0
    while not game.gameOver and turns < max_turns:
        p = game.playersGo
        b_prior = public_belief_prior(game, p)
        encoded = encode_game(game, p, b_prior)
        mask = action_mask(game)
        b_target, b_mask = belief_targets(game, p)
        action = int(player_policy[p](game, p))
        samples.append(
            {
                "card_feats": encoded["card_feats"],
                "history_feats": encoded["history_feats"],
                "global_feats": encoded["global_feats"],
                "action_mask": mask,
                "action": action,
                "player": p,
                "belief_target": b_target,
                "belief_mask": b_mask,
            }
        )
        apply_action(game, action)
        turns += 1

    rewards = np.array(game.rewards, dtype=np.float32) if game.gameOver else _unfinished_rewards(game)
    for sample in samples:
        sample["value_target"] = np.tanh(float(rewards[sample["player"] - 1]) / float(value_scale))
    return samples, rewards


class Replay:
    def __init__(self, capacity=200000):
        self.capacity = int(capacity)
        self.items = []

    def add_many(self, items):
        self.items.extend(items)
        if len(self.items) > self.capacity:
            self.items = self.items[-self.capacity :]

    def sample(self, batch_size):
        idx = np.random.choice(len(self.items), size=min(batch_size, len(self.items)), replace=False)
        return [self.items[int(i)] for i in idx]

    def __len__(self):
        return len(self.items)
