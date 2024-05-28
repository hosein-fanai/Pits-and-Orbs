import gym

import numpy as np

try:
    from environment.pits_and_orbs_env import PitsAndOrbsEnv
except:
    import sys
    import os

    current_directory = os.path.dirname(os.path.realpath(__file__))
    parent_directory = os.path.dirname(current_directory)
    sys.path.append(os.path.abspath(parent_directory))

    from environment.pits_and_orbs_env import PitsAndOrbsEnv


class SelfPlayWrapper(gym.Wrapper):

    def __init__(self, env: PitsAndOrbsEnv):
        try:
            assert env.game.team_num > 1
        except AttributeError: # it's a vectorized env made by sb3
            assert env.get_attr(attr_name="game", indices=0)[0].team_num > 1

        super().__init__(env)

        self._opponent_model = None

    def set_opponent_model(self, model: any) -> None:
        self._opponent_model = model

    def step(self, action) -> tuple[np.ndarray, float, bool, dict]:
        observation, reward, done, info = self.env.step(action)

        opponent_action, _ = self._opponent_model.predict(observation)
        self.env.step(opponent_action)

        return observation, reward, done, info