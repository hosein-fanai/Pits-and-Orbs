from gym import wrappers

from pitsandorbsenv import PitsAndOrbsEnv
from onehot_wrapper import OnehotWrapper
from normalize_observation import NormalizeObservation


def make_env(size=(5, 5), orb_num=5, pit_num=5, seed=None, onehot_obs=True, norm_obs=False, num_stack=None):
    env = PitsAndOrbsEnv(size=size, orb_num=orb_num, pit_num=pit_num, seed=seed)

    if onehot_obs:
        env = OnehotWrapper(env)

    if norm_obs and not onehot_obs:
        env = NormalizeObservation(env)

    if num_stack:
        env = wrappers.FrameStack(env)

    return env