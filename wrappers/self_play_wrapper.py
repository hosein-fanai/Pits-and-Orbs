from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv

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


class SelfPlayWrapper(gym.Wrapper, DummyVecEnv): # This is only for bypassing the isinstance(env, DummyVecEnv) in the model cereation of sb3

    def __init__(self, env: PitsAndOrbsEnv): # only supports two teams
        try:
            assert env.game.team_num == 2
            self.num_envs = 1
        except AttributeError: # it's a vectorized env made by sb3
            assert env.get_attr(attr_name="game", indices=0)[0].team_num == 2
            self.num_envs = env.num_envs

        super().__init__(env)

        self._opponent_model = None

    @property
    def opponent_rewards(self) -> float:
        return self._opponent_rewards

    def set_opponent_model(self, model: any) -> None:
        self._opponent_model = model

    def reset(self) -> np.ndarray: # , seed=None, options=None
        obs = self.env.reset()

        # TODO: the prev_observation should be taken from the other team
        # team_turn = self.game.team_turn
        # self.game.team_turn = (team_turn - 1) % self._team_num
        self.prev_observation = obs
        # self.game.team_turn = team_turn

        self._opponent_rewards = 0

        return obs

    def step(self, action: int | np.ndarray) -> tuple[np.ndarray, float, bool, dict]: # TODO: to support more teams write a while loop to step for every other agents
        observation, reward, done, info = self.env.step(action)

        opponent_action, _ = self._opponent_model.predict(self.prev_observation)
        opponent_observation, opponent_reward, *_ = self.env.step(opponent_action)
        self.prev_observation = opponent_observation
        self._opponent_rewards += opponent_reward

        return observation, reward, done, info