from gym import wrappers

from environment.pits_and_orbs_env import PitsAndOrbsEnv

from wrappers.onehot_observation import OnehotObservation
from wrappers.normalize_observation import NormalizeObservation


def make_env(max_steps=30, size=(5, 5), orb_num=5, pit_num=5, seed=None, 
            onehot_obs=True, norm_obs=False, num_stack=None):
    env = PitsAndOrbsEnv(pygame_mode=False, max_steps=max_steps, size=size, orb_num=orb_num, pit_num=pit_num, seed=seed)

    if onehot_obs and not norm_obs:
        env = OnehotObservation(env)

    if norm_obs and not onehot_obs:
        env = NormalizeObservation(env)

    if num_stack:
        env = wrappers.FrameStack(env, num_stack=num_stack)

    return env