import numpy as np
import random
from collections import deque


class ReplayBuffer:
    """
    Simple circular replay buffer.
    Stores (static, history, valid_mask, policy_target, value_target, belief_target) tuples.
    """

    def __init__(self, capacity: int = 200_000):
        self.buffer: deque = deque(maxlen=capacity)

    def add(self, static, history, valid, policy_target, value_target, belief_target):
        self.buffer.append((static, history, valid, policy_target, value_target, belief_target))

    def add_episode(self, episode_data):
        for item in episode_data:
            self.add(*item)

    def sample(self, batch_size: int):
        """
        Sample a random batch.
        Returns six np.ndarrays: statics, histories, valids, policy_targets, value_targets, belief_targets.
        """
        n = min(batch_size, len(self.buffer))
        batch = random.sample(self.buffer, n)
        statics, histories, valids, policies, values, beliefs = zip(*batch)
        return (
            np.stack(statics),
            np.stack(histories),
            np.stack(valids),
            np.stack(policies),
            np.array(values, dtype=np.float32),
            np.stack(beliefs),
        )

    def __len__(self) -> int:
        return len(self.buffer)
