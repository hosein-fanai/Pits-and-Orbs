import gym
from gym import spaces

import numpy as np


class ConcatObservation(gym.ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)

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

        players_num = env.game.players_num

        position_size = 2 if players_num > 1 else 0
        position_size = position_size*max(self.env.game.size) if len(env.observation_space.get("player0_position", np.zeros((1,))).shape) > 1 else position_size

        turn_size = 1 if players_num > 1 else 0

        self.observation_space = spaces.Box(
            low=0, 
            high=1, 
            shape=(board_size+(movements_size+direction_size+1+position_size)*players_num+turn_size,), 
            dtype=np.uint8,
        )

    def observation(self, observation):
        players_num = self.env.game.players_num

        board = observation["board"].flatten()
        player_movements = [np.array(observation[f"player{i}_movements"]).flatten() for i in range(players_num)]
        player_direction = [np.array(observation[f"player{i}_direction"]).flatten() for i in range(players_num)]
        player_has_orb = [np.array([observation[f"player{i}_has_orb"]]).flatten() for i in range(players_num)]
        player_position = [observation[f"player{i}_position"].flatten() for i in range(players_num)] if players_num > 1 else []
        player_turn = [np.array(observation[f"player_turn"]).flatten()] if players_num > 1 else []

        return np.concatenate([board, *player_movements, *player_direction, *player_has_orb, *player_position, *player_turn], axis=0)