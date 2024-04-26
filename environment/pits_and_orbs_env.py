import gym
from gym import spaces

import numpy as np

from collections import OrderedDict

try:
    from game.pits_and_orbs import PitsAndOrbs
except:
    import sys
    import os

    current_directory = os.path.dirname(os.path.realpath(__file__))
    parent_directory = os.path.dirname(current_directory)
    sys.path.append(os.path.abspath(parent_directory))

    from game.pits_and_orbs import PitsAndOrbs


class PitsAndOrbsEnv(gym.Env):

    def __init__(self, max_movements=30, render_mode="rgb_array", **kwargs):
        super(PitsAndOrbsEnv, self).__init__()

        self.max_movements = max_movements

        self._render_mode = render_mode
        pygame_mode = True if render_mode == "human" else False

        self.game = PitsAndOrbs(pygame_mode=pygame_mode, **kwargs)

        players_num = self.game.players_num
        self.observation_space = spaces.Dict({
            "board": spaces.Box(low=0, high=len(self.game.CELLS)-1, shape=self.game.size, dtype=np.uint8),
            **{f"player{i}_direction": spaces.Discrete(4) for i in range(players_num)},
            **{f"player{i}_has_orb": spaces.Discrete(2) for i in range(players_num)} |
            ({f"player{i}_position": spaces.Box(low=0, high=max(self.game.size), shape=(2,), dtype=np.uint8) for i in range(players_num)} if players_num > 1 else {})
        })
        self.action_space = spaces.Discrete(len(self.game.ACTIONS))

    def reset(self, seed=None, options=None):
        obs, info = self.game.reset_game(seed=seed)

        return self._get_obs(obs)

    def step(self, action):
        player_turn = self.game.player_turn

        if self.game.players_movements[player_turn] < self.max_movements:
            obs, reward, done, info = self.game.step(action)
            observation = self._get_obs(obs)

            if self.game.players_movements[player_turn] == self.max_movements - 1:
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
                return self.game.get_obs()

    def close(self):
        self.game.close_game()

    def _get_obs(self, obs):
        players_num = self.game.players_num

        return OrderedDict(
            [("board", obs),
            *((f"player{i}_direction", self.game.players_direction[i]) for i in range(players_num)),
            *((f"player{i}_has_orb", int(self.game.players_have_orb[i])) for i in range(players_num))] + 
            ([(f"player{i}_position", np.array(self.game.players_pos[i])) for i in range(players_num)] if players_num > 1 else [])
        )

    def _get_info(self):
        return self.game.get_info()


if __name__ == "__main__":
    env = PitsAndOrbsEnv(players_num=2)
    print(env.observation_space.sample())
    print()

    obs = env.reset()
    print(obs)
    print()

    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)

        print("Action:", action)
        print(obs)
        print("Reward:", reward)
        print()

        if done:
            break