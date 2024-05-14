import gym
from gym import spaces

import numpy as np

from collections import OrderedDict


class NormalizeObservation(gym.ObservationWrapper):

    def __init__(self, env, obs_keys="all"):
        super().__init__(env)

        self._obs_keys = obs_keys

        self._team_size = self.game.team_size
        self._team_num = self.game.team_num
        self._position_board = self.game._return_board_type == "position"

        if "board" in self._obs_keys or "all" in self._obs_keys:
            board = spaces.Box(low=0., high=1., shape=self.env.game.size, dtype=np.float32)
        else:
            board = env.observation_space["board"]

        if "movements" in self._obs_keys or "all" in self._obs_keys:
            movements = {f"player{i}_movements": spaces.Box(low=0., high=1., shape=(1,), dtype=np.float32) for i in range(self._team_size)}
        else:
            movements = {f"player{i}_movements": env.observation_space[f"player{i}_movements"] for i in range(self._team_size)}

        if "direction" in self._obs_keys or "all" in self._obs_keys:
            direction = {f"player{i}_direction": spaces.Box(low=0., high=1., shape=(1,), dtype=np.float32) for i in range(self._team_size)}
        else:
            direction = {f"player{i}_direction": env.observation_space[f"player{i}_direction"] for i in range(self._team_size)}
        
        if "has_orb" in self._obs_keys or "all" in self._obs_keys:
            has_orb = {f"player{i}_has_orb": spaces.Box(low=0., high=1., shape=(1,), dtype=np.float32) for i in range(self._team_size)} 
        else:
            has_orb = {f"player{i}_has_orb": env.observation_space[f"player{i}_has_orb"] for i in range(self._team_size)}
        
        if "position" in self._obs_keys or "all" in self._obs_keys:
            condition = (self._team_size > 1) or (self._team_num > 1) or self._position_board
            obs_space = spaces.Box(low=0., high=1., shape=(2,), dtype=np.float32)
            position = {f"player{i}_position": obs_space for i in range(self._team_size)} if condition else {}
        else:
            position = {f"player{i}_position": env.observation_space[f"player{i}_position"] for i in range(self._team_size)}
        
        if "turn" in self._obs_keys or "all" in self._obs_keys:
            turn = {"player_turn": spaces.Box(low=0., high=1., shape=(1,), dtype=np.float32)} if self._team_size > 1 else {}
        else:
            turn = env.observation_space["player_turn"]

        self.observation_space = spaces.Dict({
            "board": board,
            **movements,
            **direction,
            **has_orb,
            **position,
            **turn,
        })

    def observation(self, observation):
        if "board" in self._obs_keys or "all" in self._obs_keys:
            board = ("board", observation["board"]/(len(self.env.game.CELLS)-1))
        else:
            board = ("board", observation["board"])

        if "movements" in self._obs_keys or "all" in self._obs_keys:
            movements = ((f"player{i}_movements", np.array([observation[f"player{i}_movements"]])/self.env.max_movements) for i in range(self._team_size))
        else:
            movements = ((f"player{i}_movements", observation[f"player{i}_movements"]) for i in range(self._team_size))

        if "direction" in self._obs_keys or "all" in self._obs_keys:
            direction = ((f"player{i}_direction", np.array([observation[f"player{i}_direction"]])/3) for i in range(self._team_size))
        else:
            direction = ((f"player{i}_direction", observation[f"player{i}_direction"]) for i in range(self._team_size))
        
        if "has_orb" in self._obs_keys or "all" in self._obs_keys:
            has_orb = ((f"player{i}_has_orb", np.array([observation[f"player{i}_has_orb"]])) for i in range(self._team_size))
        else:
            has_orb = ((f"player{i}_has_orb", observation[f"player{i}_has_orb"]) for i in range(self._team_size))
        
        if "position" in self._obs_keys or "all" in self._obs_keys:
            condition = (self._team_size > 1) or (self._team_num > 1) or self._position_board
            position = [(f"player{i}_position", observation[f"player{i}_position"]/np.array(self.env.game.size)) for i in range(self._team_size)] if condition else []
        else:
            position = ((f"player{i}_position", observation[f"player{i}_position"]) for i in range(self._team_size))

        if "turn" in self._obs_keys or "all" in self._obs_keys:
            turn = ([("player_turn", np.array([observation["player_turn"]]))] if self._team_size > 1 else [])
        else:
            turn = ("player_turn", observation["player_turn"])

        return OrderedDict([
            board,
            *movements,
            *direction,
            *has_orb,
            position,
            turn,
        ])