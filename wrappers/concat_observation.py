import gym

import numpy as np


class ConcatObservation(gym.ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)

        self.observation_space = gym.spaces.Box(
            low=0, high=1,                                 
            shape=(env.game.size[0]*env.game.size[1]*len(env.game.CELLS)+4+1,), 
            dtype=np.uint8
        )

    def observation(self, observation):
        board = observation["board"].flatten()
        player_direction = np.array(observation["player_direction"]).flatten()
        player_has_orb = np.array([observation["player_has_orb"]]).flatten()

        return np.concatenate([board, player_direction, player_has_orb], axis=0)