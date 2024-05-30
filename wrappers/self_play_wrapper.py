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

    def __init__(self, env: PitsAndOrbsEnv): # only supports two teams
        try:
            assert env.game.team_num == 2
        except AttributeError: # it's a vectorized env made by sb3
            assert env.get_attr(attr_name="game", indices=0)[0].team_num == 2

        super().__init__(env)

        self._opponent_model = None

    def set_opponent_model(self, model: any) -> None:
        self._opponent_model = model

    def reset(self): # , seed=None, options=None
        obs = self.env.reset()

        # TODO: the prev_observation should be taken from the other team
        # team_turn = self.game.team_turn
        # self.game.team_turn = (team_turn - 1) % self._team_num
        self.prev_observation = obs
        # self.game.team_turn = team_turn

        return obs

    def step(self, action: int | np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        observation, reward, done, info = self.env.step(action)

        opponent_action, _ = self._opponent_model.predict(self.prev_observation)
        opponent_observation, *_ = self.env.step(opponent_action)
        self.prev_observation = opponent_observation

        return observation, reward, done, info