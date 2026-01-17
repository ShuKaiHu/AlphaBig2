import numpy as np


def _random_action(action_mask):
    valid = np.flatnonzero(action_mask)
    if valid.size == 0:
        return 0
    return int(np.random.choice(valid))


def _min_index_action(action_mask):
    valid = np.flatnonzero(action_mask)
    if valid.size == 0:
        return 0
    return int(valid[0])


def _max_index_action(action_mask):
    valid = np.flatnonzero(action_mask)
    if valid.size == 0:
        return 0
    return int(valid[-1])


def get_baseline(name):
    if name == "random":
        return _random_action
    if name == "min-index":
        return _min_index_action
    if name == "max-index":
        return _max_index_action
    raise ValueError(f"unknown baseline: {name}")
