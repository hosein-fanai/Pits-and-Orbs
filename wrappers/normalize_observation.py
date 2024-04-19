import gym

import numpy as np

from collections import OrderedDict


class NormalizeObservation(gym.ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)

        player_num = self.env.game.player_num
        self.observation_space = gym.spaces.Dict({
            "board": gym.spaces.Box(low=0, high=1, shape=self.env.game.size, dtype=np.float32)} | {
            f"player{i}_direction": gym.spaces.Box(low=0, high=1, shape=(1, ), dtype=np.float32) for i in range(player_num)} | {
            f"player{i}_has_orb": gym.spaces.Box(low=0, high=1, shape=(1, ), dtype=np.float32) for i in range(player_num)
        })

    def observation(self, observation):
        player_num = self.env.game.player_num

        return OrderedDict([
            ("board", observation["board"] / (len(self.env.game.CELLS)-1)),
            *((f"player{i}_direction", np.array([observation[f"player{i}_direction"]]) / 3) for i in range(player_num)),
            *((f"player{i}_has_orb", np.array([observation[f"player{i}_has_orb"]])) for i in range(player_num)),
        ])