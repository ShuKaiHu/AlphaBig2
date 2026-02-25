import numpy as np

import big2Game
import enumerateOptions


def _apply_action(game, action):
    if action == enumerateOptions.passInd:
        game.updateGame(-1)
    else:
        opt, n = enumerateOptions.getOptionNC(action)
        game.updateGame(opt, n)


class Node:
    def __init__(self, game, prior=0.0, parent=None, action=None):
        self.game = game
        self.prior = float(prior)
        self.parent = parent
        self.action = action
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0.0

    @property
    def q(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


class MCTS:
    def __init__(
        self,
        policy_fn,
        value_fn,
        n_simulations=50,
        c_puct=1.5,
        dirichlet_alpha=0.3,
        dirichlet_frac=0.25,
    ):
        self.policy_fn = policy_fn
        self.value_fn = value_fn
        self.n_simulations = int(n_simulations)
        self.c_puct = float(c_puct)
        self.dirichlet_alpha = float(dirichlet_alpha)
        self.dirichlet_frac = float(dirichlet_frac)

    def _terminal_value(self, game, root_player, value_scale):
        if not game.gameOver:
            return None
        raw = float(game.rewards[root_player - 1])
        return np.tanh(raw / float(value_scale))

    def _expand(self, node, root_player):
        logits = self.policy_fn(node.game, root_player)
        logits = np.array(logits, dtype=np.float32)

        avail = node.game.returnAvailableActions().astype(np.float32)
        mask = np.isfinite(big2Game.convertAvailableActions(avail.copy()))
        if not np.any(mask):
            return 0.0

        stable = logits - np.max(logits[mask])
        probs = np.zeros_like(logits, dtype=np.float32)
        probs[mask] = np.exp(stable[mask])
        s = probs[mask].sum()
        if s > 0:
            probs[mask] /= s
        else:
            probs[mask] = 1.0 / float(np.sum(mask))

        for a in np.flatnonzero(mask):
            child_game = node.game.clone()
            _apply_action(child_game, int(a))
            node.children[int(a)] = Node(
                game=child_game,
                prior=float(probs[a]),
                parent=node,
                action=int(a),
            )
        return float(self.value_fn(node.game, root_player))

    def _select_child(self, node):
        best_a = None
        best_score = -1e18
        # AlphaGo/AlphaZero PUCT:
        # score = Q(s,a) + c_puct * P(s,a) * sqrt(sum_b N(s,b)) / (1 + N(s,a))
        sqrt_n = np.sqrt(max(1, node.visit_count))
        for a, c in node.children.items():
            u = self.c_puct * c.prior * sqrt_n / (1 + c.visit_count)
            score = c.q + u
            if score > best_score:
                best_score = score
                best_a = a
        return best_a, node.children[best_a]

    def _add_root_dirichlet_noise(self, root):
        if not root.children:
            return
        actions = list(root.children.keys())
        if len(actions) == 1:
            return
        noise = np.random.dirichlet(
            [self.dirichlet_alpha] * len(actions)
        ).astype(np.float32)
        eps = self.dirichlet_frac
        for i, a in enumerate(actions):
            child = root.children[a]
            child.prior = (1.0 - eps) * child.prior + eps * float(noise[i])

    def select_action(
        self,
        game,
        root_player,
        temperature=1.0,
        value_scale=50.0,
        add_root_noise=False,
    ):
        root = Node(game=game.clone())
        _ = self._expand(root, root_player)
        if add_root_noise:
            self._add_root_dirichlet_noise(root)

        for _ in range(self.n_simulations):
            node = root
            path = [node]

            while node.children:
                _, node = self._select_child(node)
                path.append(node)

            tv = self._terminal_value(node.game, root_player, value_scale=value_scale)
            if tv is None:
                leaf_value = self._expand(node, root_player)
            else:
                leaf_value = tv

            for p in path:
                p.visit_count += 1
                p.value_sum += float(leaf_value)

        visits = np.zeros((enumerateOptions.passInd + 1,), dtype=np.float32)
        for a, c in root.children.items():
            visits[a] = c.visit_count

        valid = visits > 0
        if not np.any(valid):
            avail = game.returnAvailableActions()
            valid_actions = np.flatnonzero(avail == 1)
            if valid_actions.size == 0:
                return enumerateOptions.passInd, visits
            return int(valid_actions[0]), visits

        if temperature <= 1e-8:
            action = int(np.argmax(visits))
            return action, visits

        probs = np.zeros_like(visits)
        probs[valid] = np.power(visits[valid], 1.0 / temperature)
        probs_sum = probs.sum()
        if probs_sum <= 0:
            action = int(np.argmax(visits))
            return action, visits
        probs /= probs_sum
        action = int(np.random.choice(np.arange(len(probs)), p=probs))
        return action, visits
