import gym
from gym import spaces

import numpy as np

from collections import OrderedDict

try:
    from environment.pits_and_orbs_env import PitsAndOrbsEnv
except:
    import sys
    import os

    current_directory = os.path.dirname(os.path.realpath(__file__))
    parent_directory = os.path.dirname(current_directory)
    sys.path.append(os.path.abspath(parent_directory))

    from environment.pits_and_orbs_env import PitsAndOrbsEnv


class NormalizeObservation(gym.ObservationWrapper):

    def __init__(self, env: PitsAndOrbsEnv, obs_keys: list[str]=["all"]):
        super().__init__(env)

        self._obs_keys = obs_keys

        self._team_size = self.game.team_size
        self._team_num = self.game.team_num
        self._position_board = (self.game._return_board_type == self.game.BOARDS[1])

        self._wrap_board, self._wrap_movements, self._wrap_direction, \
            self._wrap_has_orb, self._wrap_position, self._wrap_turn = (False,) * 6

        if "board" in self._obs_keys or "all" in self._obs_keys:
            self._wrap_board = True

            if self._position_board:
                board = spaces.Box(low=0., high=1., shape=(self.game.pit_num+self.game.orb_num, 2), dtype=np.float32)
            else:
                board = spaces.Box(low=0., high=1., shape=self.game.size, dtype=np.float32)
        else:
            board = env.observation_space["board"]

        if "movements" in self._obs_keys or "all" in self._obs_keys:
            self._wrap_movements = True
        
            movements = {f"player{i}_movements": spaces.Box(low=0., high=1., shape=(1,), dtype=np.float32) for i in range(self._team_size)}
        else:
            movements = {f"player{i}_movements": env.observation_space[f"player{i}_movements"] for i in range(self._team_size)}

        if "direction" in self._obs_keys or "all" in self._obs_keys:
            self._wrap_direction = True

            direction = {f"player{i}_direction": spaces.Box(low=0., high=1., shape=(1,), dtype=np.float32) for i in range(self._team_size)}
        else:
            direction = {f"player{i}_direction": env.observation_space[f"player{i}_direction"] for i in range(self._team_size)}
        
        if "has_orb" in self._obs_keys or "all" in self._obs_keys:
            self._wrap_has_orb = True

            has_orb = {f"player{i}_has_orb": spaces.Box(low=0., high=1., shape=(1,), dtype=np.float32) for i in range(self._team_size)} 
        else:
            has_orb = {f"player{i}_has_orb": env.observation_space[f"player{i}_has_orb"] for i in range(self._team_size)}
        
        self._pos_condition = (self._team_size > 1) or (self._team_num > 1) or self._position_board
        if "position" in self._obs_keys or "all" in self._obs_keys:
            self._wrap_position = True

            obs_space = spaces.Box(low=0., high=1., shape=(2,), dtype=np.float32)
            position = {f"player{i}_position": obs_space for i in range(self._team_size)} if self._pos_condition else {}
        else:
            position = {f"player{i}_position": env.observation_space[f"player{i}_position"] for i in range(self._team_size)} if self._pos_condition else {}
        
        if "turn" in self._obs_keys or "all" in self._obs_keys:
            self._wrap_turn = True

            turn = {"player_turn": spaces.Box(low=0., high=1., shape=(1,), dtype=np.float32)} if self._team_size > 1 else {}
        else:
            turn = {"player_turn": env.observation_space["player_turn"]} if self._team_size > 1 else {}

        self.observation_space = spaces.Dict({
            "board": board,
            **movements,
            **direction,
            **has_orb,
            **position,
            **turn,
        })

    def observation(self, observation: OrderedDict):
        if self._wrap_board:
            if self._position_board:
                board = ("board", np.array(observation["board"], dtype=np.float32)/max(self.game.size))
            else:
                board = ("board", np.array(observation["board"], dtype=np.float32)/(len(self.game.CELLS)-1))
        else:
            board = ("board", observation["board"])

        if self._wrap_movements:
            movements = ((f"player{i}_movements", np.array([observation[f"player{i}_movements"]], dtype=np.float32)/self.max_movements) for i in range(self._team_size))
        else:
            movements = ((f"player{i}_movements", observation[f"player{i}_movements"]) for i in range(self._team_size))

        if self._wrap_direction:
            direction = ((f"player{i}_direction", np.array([observation[f"player{i}_direction"]], dtype=np.float32)/3) for i in range(self._team_size))
        else:
            direction = ((f"player{i}_direction", observation[f"player{i}_direction"]) for i in range(self._team_size))
        
        if self._wrap_has_orb:
            has_orb = ((f"player{i}_has_orb", np.array([observation[f"player{i}_has_orb"]], dtype=np.float32)) for i in range(self._team_size))
        else:
            has_orb = ((f"player{i}_has_orb", observation[f"player{i}_has_orb"]) for i in range(self._team_size))
        
        if self._wrap_position:
            size_array = np.array(self.game.size, dtype=np.float32)
            position = ((f"player{i}_position", np.array(observation[f"player{i}_position"], dtype=np.float32)/size_array) for i in range(self._team_size)) if self._pos_condition else ()
        else:
            position = ((f"player{i}_position", observation[f"player{i}_position"]) for i in range(self._team_size)) if self._pos_condition else ()

        if self._wrap_turn:
            turn = (("player_turn", np.array([observation["player_turn"]], dtype=np.float32)),) if self._team_size > 1 else ()
        else:
            turn = (("player_turn", observation["player_turn"]),) if self._team_size > 1 else ()

        return OrderedDict([
            board,
            *movements,
            *direction,
            *has_orb,
            *position,
            *turn,
        ])


if __name__ == "__main__":
    env = PitsAndOrbsEnv(size=7, player_num=2, team_num=1, return_board_type="positions")
    env = NormalizeObservation(env, obs_keys=["all"])

    print()
    print("---Sampled observation space from gym api:")
    print(env.observation_space.sample())
    print()

    obs = env.reset()
    print("---Returned observation from reset function:")
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