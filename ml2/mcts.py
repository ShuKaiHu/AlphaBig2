import numpy as np
import enumerateOptions


def _softmax_masked(logits, mask):
    masked = np.where(mask > 0, logits, -1e9)
    max_val = np.max(masked)
    exp = np.exp(masked - max_val) * mask
    total = np.sum(exp)
    if total <= 0:
        return mask / np.sum(mask)
    return exp / total


def _apply_action(game, action):
    if action == enumerateOptions.passInd:
        game.updateGame(-1)
        return
    opt, n_cards = enumerateOptions.getOptionNC(action)
    game.updateGame(opt, n_cards)


class TreeNode:
    def __init__(self, game, prior=0.0):
        self.game = game
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.children = {}

    @property
    def value(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


class MCTS:
    def __init__(self, policy_value_fn, n_simulations=50, c_puct=1.5):
        self.policy_value_fn = policy_value_fn
        self.n_simulations = n_simulations
        self.c_puct = c_puct

    def _select_child(self, node):
        best_score = -1e9
        best_action = None
        best_child = None
        total_visits = max(1, node.visit_count)
        for action, child in node.children.items():
            prior = child.prior
            u = self.c_puct * prior * np.sqrt(total_visits) / (1 + child.visit_count)
            score = child.value + u
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        return best_action, best_child

    def _expand(self, node, perspective_player):
        logits, value = self.policy_value_fn(node.game, perspective_player)
        avail = node.game.returnAvailableActions().astype(np.float32)
        priors = _softmax_masked(logits, avail)
        for action in np.flatnonzero(avail == 1):
            child_game = node.game.clone()
            _apply_action(child_game, int(action))
            node.children[int(action)] = TreeNode(child_game, prior=float(priors[action]))
        return float(value)

    def _terminal_value(self, game, perspective_player):
        if not getattr(game, "gameOver", 0):
            return None
        reward = float(game.rewards[perspective_player - 1])
        if reward > 0:
            return 1.0
        if reward < 0:
            return -1.0
        return 0.0

    def search(self, game, perspective_player):
        root = TreeNode(game.clone())
        for _ in range(self.n_simulations):
            node = root
            path = [node]
            while node.children:
                _, node = self._select_child(node)
                path.append(node)
            terminal_value = self._terminal_value(node.game, perspective_player)
            if terminal_value is None:
                leaf_value = self._expand(node, perspective_player)
            else:
                leaf_value = terminal_value
            for n in path:
                n.visit_count += 1
                n.value_sum += leaf_value
        visits = np.zeros((enumerateOptions.passInd + 1,), dtype=np.float32)
        for action, child in root.children.items():
            visits[action] = child.visit_count
        return visits

    def select_action(self, game, perspective_player, temperature=1.0):
        visits = self.search(game, perspective_player)
        if temperature <= 0:
            return int(np.argmax(visits)), visits
        probs = visits ** (1.0 / temperature)
        probs = probs / np.sum(probs)
        action = int(np.random.choice(len(probs), p=probs))
        return action, probs
