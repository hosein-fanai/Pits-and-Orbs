import gym

import numpy as np

from collections import OrderedDict

from pitsandorbs import PitsAndOrbs


class PitsAndOrbsEnv(gym.Env):
    MAX_STEPS = 30

    def __init__(self, **kwargs):
        super(PitsAndOrbsEnv, self).__init__()

        self.game = PitsAndOrbs(**kwargs)

        self.observation_space = gym.spaces.Dict({
            "board": gym.spaces.Box(low=0, high=10, shape=(3, 3), dtype=np.uint8),
            "player_direction": gym.spaces.Discrete(4),
            "player_has_orb": gym.spaces.Discrete(2),
            # "cells_with_multiple_orbs": gym.spaces.Box(, , shape=, ),
        })
        self.action_space = gym.spaces.Discrete(len(self.game.ACTIONS))

    def reset(self, seed=None, options=None):
        obs, info = self.game.reset(seed=seed)

        return self._get_obs(obs)

    def step(self, action):
        if self.game.player_movements < PitsAndOrbsEnv.MAX_STEPS:
            obs, reward, done, info = self.game.step(action)
            observation = self._get_obs(obs)

            if self.game.player_movements == PitsAndOrbsEnv.MAX_STEPS - 1:
                done = True # truncated

            return observation, reward, done, info

        observation = self.reset()
        reward = 0
        done = False
        info = self.game.get_info()

        return observation, reward, done, info

    def render(self, mode="human"):
        if mode == "human":
            self.game.show_board()
        elif mode == "rgb_array":
            return self.game.board_state.copy()

    def _get_obs(self, obs):
        return OrderedDict([
            ("board", obs),
            ("player_direction", self.game.player_direction),
            ("player_has_orb", int(self.game.player_has_orb)),
        ])

    def _get_info(self):
        return self.game.get_info()


