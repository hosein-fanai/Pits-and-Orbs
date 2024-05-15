from gym.wrappers import FrameStack

from environment.pits_and_orbs_env import PitsAndOrbsEnv

from wrappers.steps_limit import StepsLimit
from wrappers.onehot_observation import OnehotObservation
from wrappers.normalize_observation import NormalizeObservation
from wrappers.concat_observation import ConcatObservation


OBS_TYPES = ["board", "movements", "direction", "has_orb", "position", "turn"] # Equales to ["all"]
OBS_TYPES_WITHOUT_MOVEMENTS = OBS_TYPES.copy(); OBS_TYPES_WITHOUT_MOVEMENTS.remove("movements")


def make_env(render_mode="rgb_array", max_movements=30, 
            size=(5, 5), orb_num=5, pit_num=5, player_num=1, team_num=1, 
            return_obs_type="partial obs", return_board_type="array", 
            reward_function_type='0', reward_function=None, seed=None,
            max_steps=None, punish_on_limit=False, onehot_obs=OBS_TYPES_WITHOUT_MOVEMENTS, 
            norm_obs=["movements"], concat_obs=["all"], num_stack=None):
    env = PitsAndOrbsEnv(render_mode=render_mode, pygame_with_help=False, max_movements=max_movements,
                        size=size, orb_num=orb_num, pit_num=pit_num, player_num=player_num, team_num=team_num, 
                        return_obs_type=return_obs_type, return_board_type=return_board_type, 
                        reward_function_type=reward_function_type, reward_function=reward_function, seed=seed)

    if max_steps is not None:
        env = StepsLimit(env, max_steps=max_steps, punish_on_limit=punish_on_limit)

    if onehot_obs is not None:
        env = OnehotObservation(env, obs_keys=onehot_obs)

    if norm_obs is not None:
        env = NormalizeObservation(env, obs_keys=norm_obs)

    if concat_obs is not None:
        env = ConcatObservation(env)

    if num_stack is not None:
        env = FrameStack(env, num_stack=num_stack)

    return env


if __name__ == "__main__":
    print()

    env = make_env(players_num=2)
    print("Sampled obs from gym api:", env.observation_space.sample())
    print()

    obs = env.reset()
    print("Generated obs from reset function:", obs)
    print()

    print("Sampled obs from gym api's shape", env.observation_space.shape)
    print()

    print("Generated obs from reset function's shape", obs.shape)
    print()