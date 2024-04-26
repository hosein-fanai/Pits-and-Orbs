from gym import wrappers

from environment.pits_and_orbs_env import PitsAndOrbsEnv

from wrappers.onehot_observation import OnehotObservation
from wrappers.normalize_observation import NormalizeObservation
from wrappers.concat_observation import ConcatObservation


def make_env(render_mode="rgb_array", max_movements=30, return_obs_type="partial obs",
            reward_function_type='0', reward_function=None, size=(5, 5), orb_num=5, 
            pit_num=5, players_num=1, seed=None, onehot_obs=True, norm_obs=False, 
            num_stack=None):
    env = PitsAndOrbsEnv(render_mode=render_mode, pygame_with_help=False, 
                        max_movements=max_movements, return_obs_type=return_obs_type, 
                        reward_function_type=reward_function_type, reward_function=reward_function,
                        size=size, orb_num=orb_num, pit_num=pit_num, players_num=players_num, 
                        seed=seed)

    if onehot_obs and not norm_obs:
        env = OnehotObservation(env)

    if norm_obs and not onehot_obs:
        env = NormalizeObservation(env)

    env = ConcatObservation(env)

    if num_stack:
        env = wrappers.FrameStack(env, num_stack=num_stack)

    return env


if __name__ == "__main__":
    print()

    env = make_env()
    print("Sampled obs from gym api:", env.observation_space.sample())
    print()

    obs = env.reset()
    print("Generated obs from reset function:", obs)
    print()

    print("Sampled obs from gym api's shape", env.observation_space.shape)
    print()

    print("Generated obs from reset function's shape", obs.shape)
    print()