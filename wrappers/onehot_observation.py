import gym

import numpy as np

from collections import OrderedDict


class OnehotObservation(gym.ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)

        self.observation_space = gym.spaces.Dict({
            "board": gym.spaces.Box(low=0, high=1, shape=(*self.env.game.size, len(self.env.game.CELLS)), dtype=np.uint8),
            "player_direction": gym.spaces.Box(low=0, high=1, shape=(4,), dtype=np.uint8),
            "player_has_orb": gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.uint8),
        })

    def observation(self, observation):
        onehot_board = self._onehot(observation["board"], depth=len(self.env.game.CELLS))
        onehot_player_direction = self._onehot(np.array(observation["player_direction"]), depth=4)
        onehot_player_has_orb = np.array([observation["player_has_orb"]])

        return OrderedDict([
            ("board", onehot_board),
            ("player_direction", onehot_player_direction),
            ("player_has_orb", onehot_player_has_orb),
        ])

    def _onehot(self, arr, depth):
        arr = arr.flatten()
        output = np.zeros((arr.shape[0], depth), dtype=np.float32)
        for i, item in enumerate(arr):
            output[i, item] = 1

        return output