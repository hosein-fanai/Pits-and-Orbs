import gym
from gym import spaces

import numpy as np

from collections import OrderedDict


class NormalizeObservation(gym.ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)

        players_num = self.env.game.players_num
        self.observation_space = spaces.Dict({
            "board": spaces.Box(low=0., high=1., shape=self.env.game.size, dtype=np.float32), 
            **{f"player{i}_direction": spaces.Box(low=0., high=1., shape=(1,), dtype=np.float32) for i in range(players_num)}, 
            **{f"player{i}_has_orb": spaces.Box(low=0., high=1., shape=(1,), dtype=np.float32) for i in range(players_num)} | 
            ({f"player{i}_position": spaces.Box(low=0., high=1., shape=(2,), dtype=np.float32) for i in range(players_num)} if players_num > 1 else {})
        })

    def observation(self, observation):
        players_num = self.env.game.players_num

        return OrderedDict([
            ("board", observation["board"] / (len(self.env.game.CELLS)-1)),
            *((f"player{i}_direction", np.array([observation[f"player{i}_direction"]]) / 3) for i in range(players_num)),
            *((f"player{i}_has_orb", np.array([observation[f"player{i}_has_orb"]])) for i in range(players_num))] + 
            ([(f"player{i}_position", observation[f"player{i}_position"] / np.array(self.env.game.size)) for i in range(players_num)] if players_num > 1 else [])
        )