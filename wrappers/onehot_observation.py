import gym
from gym import spaces

import numpy as np

from collections import OrderedDict


class OnehotObservation(gym.ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)

        team_size = self.game.team_size
        team_num = self.game.team_num
        position_board = self.game._return_board_type == "position"
        self.observation_space = spaces.Dict({
            "board": spaces.Box(low=0, high=1, shape=(*self.env.game.size, len(self.env.game.CELLS)), dtype=np.uint8), 
            **{f"player{i}_direction": spaces.Box(low=0, high=1, shape=(4,), dtype=np.uint8) for i in range(team_size)}, 
            **{f"player{i}_has_orb": spaces.Box(low=0, high=1, shape=(1,), dtype=np.uint8) for i in range(team_size)} | 
            ({f"player{i}_position": spaces.Box(low=0, high=1, shape=(2, max(self.env.game.size)), dtype=np.uint8) for i in range(team_size)} if (team_size > 1) or (team_num > 1) or position_board else {}) | # TODO: make it support rectangular grids
            ({"player_turn": spaces.Box(low=0, high=1, shape=(1,), dtype=np.uint8)} if team_size > 1 else {})
        })

    def observation(self, observation):
        team_size = self.env.game.team_size
        team_num = self.game.team_num
        position_board = self.game._return_board_type == "position"

        return OrderedDict([
            ("board", self._onehot(observation["board"], depth=len(self.env.game.CELLS))), 
            *((f"player{i}_direction", self._onehot(np.array(observation[f"player{i}_direction"]), depth=4)) for i in range(team_size)), 
            *((f"player{i}_has_orb", np.array([observation[f"player{i}_has_orb"]])) for i in range(team_size))] + 
            ([(f"player{i}_position", self._onehot(observation[f"player{i}_position"], depth=max(self.env.game.size))) for i in range(team_size)] if (team_size > 1) or (team_num > 1) or position_board else []) + 
            ([("player_turn", np.array([observation["player_turn"]]))] if team_size > 1 else [])
        )

    def _onehot(self, arr, depth):
        arr = arr.flatten()
        output = np.zeros((arr.shape[0], depth), dtype=np.uint8)
        for i, item in enumerate(arr):
            output[i, item] = 1

        return output