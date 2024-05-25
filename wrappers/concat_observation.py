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


class ConcatObservation(gym.ObservationWrapper):

    def __init__(self, env: PitsAndOrbsEnv, obs_keys: list[str]=["all"]):
        super().__init__(env)

        self._obs_keys = obs_keys

        self._team_size = self.game.team_size
        self._team_num = self.game.team_num
        self._position_board = (self.game._return_board_type == self.game.BOARDS[1])

        board_size = 1
        for dim in env.observation_space["board"].shape:
            board_size *= dim

        try:
            movements_size = env.observation_space["player0_movements"].shape[0]
        except:
            movements_size = 1

        try:
            direction_size = env.observation_space["player0_direction"].shape[0]
        except:
            direction_size = 1

        has_orb_size = 1

        position_size = 2 if self._team_size > 1 else 0
        position_size = position_size*max(self.game.size) if len(env.observation_space.get("player0_position", np.zeros((1,))).shape) > 1 else position_size

        turn_size = 1 if self._team_size > 1 else 0

        self._wrap_board, self._wrap_movements, self._wrap_direction, \
            self._wrap_has_orb, self._wrap_position, self._wrap_turn = (True,) * 6

        if "board" not in self._obs_keys and "all" not in self._obs_keys:
            self._wrap_board = False

            board_size = 0

        if "movements" not in self._obs_keys and "all" not in self._obs_keys:
            self._wrap_movements = False

            movements_size = 0

        if "direction" not in self._obs_keys and "all" not in self._obs_keys:
            self._wrap_direction = False

            direction_size = 0

        if "has_orb" not in self._obs_keys and "all" not in self._obs_keys:
            self._wrap_has_orb = False

            has_orb_size = 0

        if "position" not in self._obs_keys and "all" not in self._obs_keys:
            self._wrap_position = False

            position_size = 0

        if "turn" not in self._obs_keys and "all" not in self._obs_keys: 
            self._wrap_turn = False

            turn_size = 0

        self.observation_space = spaces.Box(
            low=0., 
            high=1., 
            shape=(board_size+(movements_size+direction_size+has_orb_size+position_size)*self._team_size+turn_size,), 
            dtype=np.float32,
        )

    def observation(self, observation: OrderedDict):
        if self._wrap_board:
            board = observation["board"].flatten()
        else:
            board = []

        if self._wrap_movements:
            player_movements = [np.array(observation[f"player{i}_movements"]).flatten() for i in range(self._team_size)]
        else:
            player_movements = []

        if self._wrap_direction:
            player_direction = [np.array(observation[f"player{i}_direction"]).flatten() for i in range(self._team_size)]
        else:
            player_direction = []

        if self._wrap_has_orb:
            player_has_orb = [np.array([observation[f"player{i}_has_orb"]]).flatten() for i in range(self._team_size)]
        else:
            player_has_orb = []

        if self._wrap_position:
            condition = (self._team_size > 1) or (self._team_num > 1) or self._position_board
            player_position = [observation[f"player{i}_position"].flatten() for i in range(self._team_size)] if condition else []
        else:
            player_position = []

        if self._wrap_turn:
            player_turn = [np.array(observation[f"player_turn"]).flatten()] if self._team_size > 1 else []
        else:
            player_turn = []

        return np.concatenate([board, *player_movements, *player_direction, *player_has_orb, *player_position, *player_turn], axis=0)


if __name__ == "__main__":
    env = PitsAndOrbsEnv(size=7, player_num=2, team_num=1, return_board_type="positions")
    env = ConcatObservation(env, obs_keys=["all"])

    print()
    print("---Sampled observation space from gym api:")
    print(env.observation_space.sample())
    print()

    obs = env.reset()
    print("---Returned observation from reset function:")
    print(obs)
    print()

    print("---Sampled obs from gym api's shape", env.observation_space.shape)
    print()

    print("---Generated obs from reset function's shape", obs.shape)
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