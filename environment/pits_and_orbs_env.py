import gym

import numpy as np

from collections import OrderedDict

from game.pits_and_orbs import PitsAndOrbs


class PitsAndOrbsEnv(gym.Env):

    def __init__(self, max_movements=30, render_mode="rgb_array", **kwargs):
        super(PitsAndOrbsEnv, self).__init__()

        self.max_movements = max_movements

        self._render_mode = render_mode
        pygame_mode = True if render_mode == "human" else False

        self.game = PitsAndOrbs(pygame_mode=pygame_mode, **kwargs)

        self.observation_space = gym.spaces.Dict({
            "board": gym.spaces.Box(low=0, high=len(self.game.CELLS)-1, shape=(3, 3), dtype=np.uint8),
            "player_direction": gym.spaces.Discrete(4),
            "player_has_orb": gym.spaces.Discrete(2),
        })
        self.action_space = gym.spaces.Discrete(len(self.game.ACTIONS))

    def reset(self, seed=None, options=None):
        obs, info = self.game.reset_game(seed=seed)

        return self._get_obs(obs)

    def step(self, action):
        if self.game.player_movements < self.max_movements:
            obs, reward, done, info = self.game.step(action)
            observation = self._get_obs(obs)

            if self.game.player_movements == self.max_movements - 1:
                done = True # truncated

            return observation, reward, done, info

        observation = self.reset()
        reward = 0
        done = False
        info = self.game.get_info()

        return observation, reward, done, info

    def render(self, mode="human"):
        assert not(self._render_mode == "rgb_array" and mode == "human")

        if mode == "human":
            self.game.show_board()
        elif mode == "rgb_array":
            if self.game._pygame_mode:
                self.game._update_screen()
                return self.game.get_frame()
            else:
                return self.game.get_partial_obs_with_mem(self._get_info())

    def close(self):
        self.game.close_game()

    def _get_obs(self, obs):
        return OrderedDict([
            ("board", obs),
            ("player_direction", self.game.player_direction),
            ("player_has_orb", int(self.game.player_has_orb)),
        ])

    def _get_info(self):
        return self.game.get_info()


# if __name__ == "__main__":
    env = PitsAndOrbsEnv()
    obs = env.reset()

    for _ in range(1000):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)

        if done:
            break