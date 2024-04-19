import gym

import numpy as np

from collections import OrderedDict


class OnehotObservation(gym.ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)

        player_num = self.env.game.player_num
        self.observation_space = gym.spaces.Dict({
            "board": gym.spaces.Box(low=0, high=1, shape=(*self.env.game.size, len(self.env.game.CELLS)), dtype=np.uint8)} | {
            f"player{i}_direction": gym.spaces.Box(low=0, high=1, shape=(4,), dtype=np.uint8) for i in range(player_num)} | {
            f"player{i}_has_orb": gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.uint8) for i in range(player_num)
        })

    def observation(self, observation):
        player_num = self.env.game.player_num

        return OrderedDict([
            ("board", self._onehot(observation["board"], depth=len(self.env.game.CELLS))),
            *((f"player{i}_direction", self._onehot(np.array(observation[f"player{i}_direction"]), depth=4)) for i in range(player_num)),
            *((f"player{i}_has_orb", np.array([observation[f"player{i}_has_orb"]])) for i in range(player_num)),
        ])

    def _onehot(self, arr, depth):
        arr = arr.flatten()
        output = np.zeros((arr.shape[0], depth), dtype=np.float32)
        for i, item in enumerate(arr):
            output[i, item] = 1

        return output