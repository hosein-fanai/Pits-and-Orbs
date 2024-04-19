import gym

import numpy as np

from collections import OrderedDict


class NormalizeObservation(gym.ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)

        self.observation_space = gym.spaces.Dict({
            "board": gym.spaces.Box(low=0, high=1, shape=self.env.game.size, dtype=np.float32),
            "player_direction": gym.spaces.Box(low=0, high=1, shape=(1, ), dtype=np.float32),
            "player_has_orb": gym.spaces.Box(low=0, high=1, shape=(1, ), dtype=np.float32),
        })

    def observation(self, observation):
        norm_board = observation["board"] / len(self.env.game.CELLS)
        norm_player_direction = np.array(observation["player_direction"]) / 4
        norm_player_has_orb = observation["player_has_orb"]

        return OrderedDict([
            ("board", norm_board),
            ("player_direction", norm_player_direction),
            ("player_has_orb", norm_player_has_orb),
        ])