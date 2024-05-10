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

        team_size = self.game.team_size
        team_num = self.game.team_num
        position_board = self.game._return_board_type == "position"
        self.observation_space = spaces.Dict({
            "board": spaces.Box(low=0, high=len(self.game.CELLS)-1, shape=self.game.size, dtype=np.uint8),
            # **{f"player{i}_movements": spaces.Discrete(self.max_movements) for i in range(players_num)}, 
            **{f"player{i}_direction": spaces.Discrete(4) for i in range(team_size)},
            **{f"player{i}_has_orb": spaces.Discrete(2) for i in range(team_size)} |
            ({f"player{i}_position": spaces.Box(low=0, high=max(self.game.size), shape=(2,), dtype=np.uint8) for i in range(team_size)} if (team_size > 1) or (team_num > 1) or position_board else {}) | 
            ({"player_turn": spaces.Discrete(2)} if team_size > 1 else {}) | 
            ({"team_turn": spaces.Discrete(2)} if team_num > 1 else {})
        })

        if self.game.team_num > 1:
            action_n = len(self.game.ACTIONS)
        else:
            action_n = len(self.game.ACTIONS) - 1
        self.action_space = spaces.Discrete(action_n)

    def _get_final_step_reward(self, done):
        filled_all_pits = self.game._calc_filled_pits() == self.game.pit_num

        if done and filled_all_pits:
            reward_type = "episode is done successfully"
        elif done and not filled_all_pits:
            reward_type = "episode is done unsuccessfully"
        else:
            reward_type = "episode is not done"
        reward = self.game._reward_function(flag=reward_type) 

        return reward

    def _all_players_used_max_moves(self):
        flag = True
        for team in self.game.teams:
            for player in team.players:
                flag = flag and (player.movements == self.max_movements)

        return flag

    def _get_obs(self, obs):
        team_size = self.game.team_size
        team_num = self.game.team_num
        team = self.game.current_team
        position_board = self.game._return_board_type == "position"

        return OrderedDict(
            [("board", obs), 
            # *((f"player{i}_movements", self.game.players_movements[i]) for i in range(players_num)), 
            *((f"player{i}_direction", player.direction) for i, player in enumerate(team.players)), 
            *((f"player{i}_has_orb", int(player.has_orb)) for i, player in enumerate(team.players))] + 
            ([(f"player{i}_position", np.array(player.position)) for i, player in enumerate(team.players)] if (team_size > 1) or (team_num > 1) or position_board else []) + 
            ([(f"player_turn", team.player_turn)] if team_size > 1 else []) + 
            ([(f"team_turn", self.game.team_turn)] if team_num > 1 else [])
        )

    def _get_info(self):
        return self.game._get_info()

    def reset(self, seed=None, options=None):
        obs, info = self.game.reset_game(seed=seed)

        return self._get_obs(obs)

    def step(self, action):
        # this is for when the player still has moves to play
        if self.game.current_player.movements < self.max_movements:
            raw_obs, reward, done, info = self.game.step_game(action)
            observation = self._get_obs(raw_obs)
            
            flag = self._all_players_used_max_moves()
            done = done or flag # truncates if all the players have used their max moves

            reward += self._get_final_step_reward(done)

            return observation, reward, done, info

        # this is for when the player doesn't have any moves to play, but others may have or have not
        raw_obs = self.game._get_observation()
        observation = self._get_obs(raw_obs)
        reward = 0.
        done = self._all_players_used_max_moves() 
        info = self.game._get_info()

        reward += self._get_final_step_reward(done)

        self.game._change_team_and_player_turn()

        return observation, reward, done, info

    def render(self, mode="human"):
        assert not(self._render_mode == "rgb_array" and mode == "human")

        if mode == "human":
            self.game.render_game()
        elif mode == "rgb_array":
            if self.game._pygame_mode:
                self.game._update_screen()
                return self.game.get_frame()
            else:
                return self.game._get_observation()

    def close(self):
        self.game.close_game()


if __name__ == "__main__":
    env = PitsAndOrbsEnv(size=(7, 7), player_num=2, team_num=2)

    print()
    print("---Sampled observation space from gym api:")
    print(env.observation_space.sample())
    print()

    obs = env.reset()
    print(obs)
    print()

    for _ in range(10):
        action = env.action_space.sample()
        # action = int(input("Current Action: "))
        obs, reward, done, info = env.step(action)

        print("Action:", action)
        print(obs)
        print("Reward:", reward)
        print()

        if done:
            break