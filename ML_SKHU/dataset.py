import random
from collections import deque


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.belief = deque(maxlen=capacity)
        self.policy = deque(maxlen=capacity)

    def add_belief(self, x, target, mask):
        self.belief.append((x, target, mask))

    def add_policy(self, x, policy_target, value_target, action_mask):
        self.policy.append((x, policy_target, value_target, action_mask))

    def sample_belief(self, batch_size):
        return random.sample(self.belief, min(batch_size, len(self.belief)))

    def sample_policy(self, batch_size):
        samples = random.sample(self.policy, min(batch_size, len(self.policy)))
        xs, ys, vs, masks = zip(*samples)
        return list(zip(xs, ys, vs, masks))

    def __len__(self):
        return len(self.belief), len(self.policy)
