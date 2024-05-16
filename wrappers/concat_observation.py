import gym
from gym import spaces

import numpy as np


class ConcatObservation(gym.ObservationWrapper):

    def __init__(self, env, obs_keys="all"):
        super().__init__(env)

        self._obs_keys = obs_keys

        self._team_size = self.game.team_size
        self._team_num = self.game.team_num
        self._position_board = self.game._return_board_type == "position"

        board_size = 1
        for dim in env.observation_space["board"].shape:
            board_size *= dim

        try:
            movements_size = env.observation_space["player0_movements"].shape[0]
        except:
            movements_size = 1

        try:
            direction_size = env.observation_space["player0_direction"].shape[0]
        except:
            direction_size = 1

        has_orb_size = 1

        position_size = 2 if self._team_size > 1 else 0
        position_size = position_size*max(self.env.game.size) if len(env.observation_space.get("player0_position", np.zeros((1,))).shape) > 1 else position_size

        turn_size = 1 if self._team_size > 1 else 0

        if "board" not in self._obs_keys and "all" not in self._obs_keys:
            board_size = 0

        if "movements" not in self._obs_keys and "all" not in self._obs_keys:
            movements_size = 0

        if "direction" not in self._obs_keys and "all" not in self._obs_keys:
            direction_size = 0
        
        if "has_orb" not in self._obs_keys and "all" not in self._obs_keys:
            has_orb_size = 0
        
        if "position" not in self._obs_keys and "all" not in self._obs_keys:
            position_size = 0
        
        if "turn" not in self._obs_keys and "all" not in self._obs_keys: 
            turn_size = 0

        self.observation_space = spaces.Box(
            low=0., 
            high=1., 
            shape=(board_size+(movements_size+direction_size+has_orb_size+position_size)*self._team_size+turn_size,), 
            dtype=np.float32,
        )

    def observation(self, observation):
        if "board" in self._obs_keys or "all" in self._obs_keys:
            board = observation["board"].flatten()
        else:
            board = []

        if "movements" in self._obs_keys or "all" in self._obs_keys:
            player_movements = [np.array(observation[f"player{i}_movements"]).flatten() for i in range(self._team_size)]
        else:
            player_movements = []

        if "direction" in self._obs_keys or "all" in self._obs_keys:
            player_direction = [np.array(observation[f"player{i}_direction"]).flatten() for i in range(self._team_size)]
        else:
            player_direction = []

        if "has_orb" in self._obs_keys or "all" in self._obs_keys:
            player_has_orb = [np.array([observation[f"player{i}_has_orb"]]).flatten() for i in range(self._team_size)]
        else:
            player_has_orb = []

        if "position" in self._obs_keys or "all" in self._obs_keys:
            player_position = [observation[f"player{i}_position"].flatten() for i in range(self._team_size)] if self._team_size > 1 else []
        else:
            player_position = []

        if "turn" in self._obs_keys or "all" in self._obs_keys:
            condition = (self._team_size > 1) or (self._team_num > 1) or self._position_board
            player_turn = [np.array(observation[f"player_turn"]).flatten()] if condition else []
        else:
            player_turn = []

        return np.concatenate([board, *player_movements, *player_direction, *player_has_orb, *player_position, *player_turn], axis=0)