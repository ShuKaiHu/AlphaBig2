import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import time
import numpy as np
from engine.features import encode_static, encode_history_steps


class MCTSNode:
    __slots__ = (
        "env", "player", "action", "parent", "children",
        "prior", "visit_count", "value_sum", "_terminal_value",
        "priors",   # dict {action: prior_prob} — set by _expand, enables lazy child creation
    )

    def __init__(self, env, player: int, action=None, parent=None, prior: float = 0.0):
        self.env = env          # Big2Env clone representing state AFTER this action
        self.player = player    # whose turn it is at this node
        self.action = action    # action that led here (None for root)
        self.parent = parent
        self.children: dict = {}
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self._terminal_value = None  # set when node is terminal
        self.priors = None           # set by _expand() once model runs

    @property
    def q_value(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0.0

    def is_leaf(self) -> bool:
        """True if model has NOT yet been run on this node (unexpanded)."""
        return self.priors is None

    def is_terminal(self) -> bool:
        return self._terminal_value is not None

    def ucb_score(self, parent_visits: int, c_puct: float) -> float:
        u = c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)
        return self.q_value + u


class MCTS:
    """
    PUCT-based MCTS using the Big2Net policy + value heads.

    Lazy expansion: _expand() runs the model and stores priors but does NOT
    create child nodes. _select() creates exactly ONE child per simulation
    (the lazily-chosen best action), so each simulation does only 1 env.clone()
    instead of len(valid_actions) clones. This is ~50-100× faster when the
    branching factor is large (control=1 hand).
    """

    REWARD_SCALE = 13.0  # normalise raw Big2 rewards by this before tanh

    def __init__(
        self,
        model,
        n_simulations: int = 50,
        c_puct: float = 2.0,
        dirichlet_frac: float = 0.25,
    ):
        self.model = model
        self.n_simulations = n_simulations
        self.c_puct = c_puct
        self.dirichlet_frac = dirichlet_frac

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, env, temperature: float = 1.0, time_limit: float = None):
        """
        Run MCTS from current `env` state.

        Returns:
            action: int  — selected action
            visits: np.ndarray (ACTION_SIZE,)  — visit count distribution
        """
        root = MCTSNode(env.clone(), env.current_player)
        self._expand(root)   # model forward pass; stores priors, no child cloning

        if time_limit is not None:
            deadline = time.time() + time_limit
            while time.time() < deadline:
                leaf, path = self._select(root)
                if leaf.is_terminal():
                    value = leaf._terminal_value
                elif leaf.is_leaf():
                    value = self._expand(leaf)
                else:
                    value = leaf.q_value
                self._backup(path, value, leaf.player)
        else:
            for _ in range(self.n_simulations):
                leaf, path = self._select(root)
                if leaf.is_terminal():
                    value = leaf._terminal_value
                elif leaf.is_leaf():
                    value = self._expand(leaf)
                else:
                    value = leaf.q_value
                self._backup(path, value, leaf.player)

        visits = self._visit_counts(root, env.ACTION_SIZE)
        action = self._select_action(visits, temperature)
        return action, visits

    # ── Internal ──────────────────────────────────────────────────────────────

    def _expand(self, node: MCTSNode) -> float:
        """
        Run the model on `node.env`, store priors.
        Does NOT create any child nodes (lazy expansion).
        Returns the value estimate for this node.
        """
        env = node.env
        game = env.game
        player = env.current_player
        static = encode_static(game, player)
        history = encode_history_steps(game)
        valid = env.get_valid_actions()
        probs, value = self.model.predict(static, history, valid)

        valid_actions = np.flatnonzero(valid)
        if len(valid_actions) == 0:
            node.priors = {}
            return value

        # Dirichlet noise at root — use fixed alpha=0.3 so all valid actions
        # get meaningful noise regardless of branching factor.
        if node.parent is None and self.dirichlet_frac > 0:
            alpha = 0.3
            noise = np.random.dirichlet([alpha] * len(valid_actions))
            probs = probs.copy()
            probs[valid_actions] = (
                (1 - self.dirichlet_frac) * probs[valid_actions]
                + self.dirichlet_frac * noise
            )

        node.priors = {int(a): float(probs[a]) for a in valid_actions}
        return value

    def _make_child(self, parent: MCTSNode, action: int) -> "MCTSNode":
        """
        Lazily create one child by cloning parent.env and stepping action.
        This is the ONLY place env.clone() is called per simulation.
        """
        child_env = parent.env.clone()
        rewards, done = child_env.step(action)

        if done:
            raw = float(rewards[parent.player - 1])
            term_val = float(np.tanh(raw / self.REWARD_SCALE))
            child_player = parent.player
        else:
            term_val = None
            child_player = child_env.current_player

        child = MCTSNode(
            env=child_env,
            player=child_player,
            action=action,
            parent=parent,
            prior=parent.priors[action],
        )
        if done:
            child._terminal_value = term_val
        parent.children[action] = child
        return child

    def _select(self, root: MCTSNode):
        """
        Traverse tree using PUCT until a leaf or terminal node.

        For expanded nodes (have priors), PUCT considers:
          - existing children: use standard UCB
          - unvisited actions (in priors but not children): UCB with visit=0
        When the best action is unvisited, lazily create that child and stop.
        """
        path = [root]
        node = root

        while not node.is_terminal():
            if node.is_leaf():
                # Model hasn't been run here yet → stop, caller will expand
                return node, path

            # Node is expanded: pick best action via PUCT
            N = node.visit_count
            sqrt_N = math.sqrt(max(N, 1))
            best_score = -math.inf
            best_action = None

            for a, prior in node.priors.items():
                if a in node.children:
                    score = node.children[a].ucb_score(N, self.c_puct)
                else:
                    # Unvisited child: Q=0, U = c_puct * prior * sqrt(N) / 1
                    score = self.c_puct * prior * sqrt_N
                if score > best_score:
                    best_score = score
                    best_action = a

            if best_action is None:
                break

            if best_action not in node.children:
                # Lazily create the child (1 clone per simulation)
                child = self._make_child(node, best_action)
                path.append(child)
                return child, path   # new child is always a leaf → backup

            node = node.children[best_action]
            path.append(node)

        return node, path

    def _backup(self, path, value: float, leaf_player: int):
        """
        Backup value. For nodes where it's the same player's turn,
        add +value; for opponent nodes, add -value.
        """
        for node in reversed(path):
            if node.player == leaf_player:
                node.value_sum += value
            else:
                node.value_sum -= value
            node.visit_count += 1

    @staticmethod
    def _visit_counts(root: MCTSNode, action_size: int) -> np.ndarray:
        visits = np.zeros(action_size, dtype=np.float32)
        for a, child in root.children.items():
            visits[a] = child.visit_count
        return visits

    @staticmethod
    def _select_action(visits: np.ndarray, temperature: float) -> int:
        if temperature < 1e-4:
            return int(np.argmax(visits))
        visits_t = visits ** (1.0 / temperature)
        total = visits_t.sum()
        if total == 0:
            return int(np.argmax(visits))
        probs = visits_t / total
        return int(np.random.choice(len(probs), p=probs))
