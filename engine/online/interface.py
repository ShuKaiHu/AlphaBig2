"""
Abstract interface for connecting the trained model to an online Big 2 platform.

To support a specific platform, subclass OnlineGameInterface and implement:
  - capture_game_state()
  - execute_action()
  - wait_for_my_turn()

Then call run_game_loop(model) to start playing.
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from abc import ABC, abstractmethod
import numpy as np

import enumerateOptions
PASS_IDX = enumerateOptions.passInd


class OnlineGameInterface(ABC):
    """
    Abstract bridge between the trained Big2Net and an online game platform.

    Subclasses must implement three methods:
      capture_game_state() → fills a big2Game-compatible state
      execute_action(card_ids)
      wait_for_my_turn()
    """

    def __init__(self, my_player: int = 1):
        self.my_player = my_player

    # ── Abstract methods (implement per platform) ─────────────────────────────

    @abstractmethod
    def capture_game_state(self):
        """
        Read the current game state from the screen or platform API.
        Must return a big2Game instance (or compatible object) with:
          - currentHands[self.my_player] populated (our hand)
          - cardsPlayed populated
          - playersGo, control, mustPlayClub3, etc. set correctly
        """
        ...

    @abstractmethod
    def execute_action(self, card_ids: list) -> None:
        """
        Play the given cards on the platform.
        card_ids: list of card ints (1–52). Empty list = pass.
        """
        ...

    @abstractmethod
    def wait_for_my_turn(self, timeout: float = 60.0) -> bool:
        """
        Block until it is our turn to play.
        Returns True if our turn was detected, False on timeout.
        """
        ...

    # ── Game loop ─────────────────────────────────────────────────────────────

    def run_game_loop(
        self,
        model,
        verbose: bool = True,
        time_limit: float = 3.0,
        n_simulations: int = None,
    ) -> None:
        """
        Main inference loop:
          wait → capture state → MCTS search (time_limit seconds) → execute action → repeat

        Args:
            model:        trained Big2Net
            verbose:      print debug info
            time_limit:   seconds per move (default 3.0). Set to None to use n_simulations.
            n_simulations: fixed sim count fallback when time_limit is None.
        """
        import time as _time
        from engine.features import encode_static, encode_history_steps
        from engine.env import Big2Env
        from engine.mcts import MCTS

        mcts = MCTS(model, n_simulations=n_simulations or 200)

        while True:
            if not self.wait_for_my_turn():
                if verbose:
                    print("Timeout waiting for turn, stopping.")
                break

            game = self.capture_game_state()

            # Reconstruct a Big2Env from captured state for MCTS
            env = Big2Env()
            env.game = game
            env.done = False

            valid = env.get_valid_actions()

            t0 = _time.time()
            action, visits = mcts.run(env, temperature=0.0, time_limit=time_limit)
            elapsed = _time.time() - t0
            n_sims_done = int(visits.sum())

            if valid[action] == 0:
                # Safety fallback: MCTS picked an invalid action
                masked = visits * valid
                action = int(np.argmax(masked)) if masked.sum() > 0 else PASS_IDX

            if verbose:
                from engine.features import encode_static, encode_history_steps
                static = encode_static(game, self.my_player)
                history = encode_history_steps(game)
                _, value = model.predict(static, history, valid)
                print(f"MCTS: {n_sims_done} sims in {elapsed:.2f}s | value={value:+.3f}")

            if action == PASS_IDX:
                card_ids = []
            else:
                opt, nC = enumerateOptions.getOptionNC(action)
                card_ids = self._resolve_card_ids(game, opt, nC)

            if verbose:
                print(f"Playing: {card_ids if card_ids else 'PASS'}")

            self.execute_action(card_ids)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _resolve_card_ids(self, game, opt, nC: int) -> list:
        """Convert (opt, nC) from getOptionNC back to actual card IDs."""
        hand = game.currentHands[self.my_player]
        if nC == 0:
            return []
        if nC == 1:
            return [int(hand[opt])]
        if nC == 2:
            indices = enumerateOptions.inverseTwoCardIndices[opt]
            return [int(hand[i]) for i in indices]
        if nC == 5:
            indices = enumerateOptions.inverseFiveCardIndices[opt]
            return [int(hand[i]) for i in indices]
        return []
