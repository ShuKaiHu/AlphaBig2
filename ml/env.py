import numpy as np

from big2Game import big2Game


class Big2Env:
    def __init__(self):
        self.game = big2Game()

    def reset(self):
        player, state, avail = self.game.getCurrentState()
        return self._format_state(player, state, avail)

    def step(self, action):
        acting_player = self.game.playersGo
        reward, done, info = self.game.step(int(action))
        player, state, avail = self.game.getCurrentState()
        obs = self._format_state(player, state, avail)
        out_info = {
            "acting_player": int(acting_player),
            "reward_vector": reward if done else None,
            "raw_info": info,
        }
        return obs, done, out_info

    def _format_state(self, player, state, avail):
        obs = np.asarray(state, dtype=np.float32).reshape(-1)
        avail = np.asarray(avail, dtype=np.float32).reshape(-1)
        action_mask = np.isfinite(avail)
        return {
            "player": int(player),
            "obs": obs,
            "action_mask": action_mask,
        }
