import gym
from gym import spaces

import numpy as np

from collections import OrderedDict


class OnehotObservation(gym.ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)

        players_num = self.env.game.players_num
        self.observation_space = spaces.Dict({
            "board": spaces.Box(low=0, high=1, shape=(*self.env.game.size, len(self.env.game.CELLS)), dtype=np.uint8),
            **{f"player{i}_direction": spaces.Box(low=0, high=1, shape=(4,), dtype=np.uint8) for i in range(players_num)},
            **{f"player{i}_has_orb": spaces.Box(low=0, high=1, shape=(1,), dtype=np.uint8) for i in range(players_num)} | 
            ({f"player{i}_position": spaces.Box(low=0, high=1, shape=(2, max(self.env.game.size)), dtype=np.uint8) for i in range(players_num)} if players_num > 1 else {}) # TODO: make it support rectangular grids
        })

    def observation(self, observation):
        players_num = self.env.game.players_num

        return OrderedDict([
            ("board", self._onehot(observation["board"], depth=len(self.env.game.CELLS))),
            *((f"player{i}_direction", self._onehot(np.array(observation[f"player{i}_direction"]), depth=4)) for i in range(players_num)),
            *((f"player{i}_has_orb", np.array([observation[f"player{i}_has_orb"]])) for i in range(players_num))] + 
            ([(f"player{i}_position", self._onehot(observation[f"player{i}_position"], depth=max(self.env.game.size))) for i in range(players_num)] if players_num > 1 else [])
        )

    def _onehot(self, arr, depth):
        arr = arr.flatten()
        output = np.zeros((arr.shape[0], depth), dtype=np.uint8)
        for i, item in enumerate(arr):
            output[i, item] = 1

        return output