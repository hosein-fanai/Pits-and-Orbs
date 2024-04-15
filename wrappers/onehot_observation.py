import gym

import numpy as np


class OnehotObservation(gym.ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)

        self.observation_space = gym.spaces.Box(low=0, high=1, 
                                            shape=(3*3*len(env.game.CELLS)+4+1,), 
                                            dtype=np.float32)

    def observation(self, observation):
        onehot_board = self._onehot(observation["board"], depth=len(self.env.game.CELLS)).flatten()
        onehot_player_direction = self._onehot(np.array(observation["player_direction"]), depth=4).flatten()
        onehot_player_has_orb = np.array([observation["player_has_orb"]])

        return np.concatenate([onehot_board, onehot_player_direction, onehot_player_has_orb], axis=0)

    def _onehot(self, arr, depth):
        arr = arr.flatten()
        output = np.zeros((arr.shape[0], depth), dtype=np.float32)
        for i, item in enumerate(arr):
            output[i, item] = 1

        return output


