import gym

import numpy as np


class ConcatObservation(gym.ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)

        board_size = 1
        for dim in env.observation_space["board"].shape:
            board_size *= dim

        try:
            direction_size = env.observation_space["player0_direction"].shape[0]
        except:
            direction_size = 1

        self.observation_space = gym.spaces.Box(
            low=0, high=1,                                 
            shape=(board_size+(direction_size+1)*env.game.player_num,), 
            dtype=np.uint8
        )

    def observation(self, observation):
        player_num = self.env.game.player_num

        board = observation["board"].flatten()
        player_direction = [np.array(observation[f"player{i}_direction"]).flatten() for i in range(player_num)]
        player_has_orb = [np.array([observation[f"player{i}_has_orb"]]).flatten() for i in range(player_num)]

        return np.concatenate([board, *player_direction, *player_has_orb], axis=0)