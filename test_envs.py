from stable_baselines3.common.env_checker import check_env

import gym

from environment.pits_and_orbs_env import PitsAndOrbsEnv

from game.pits_and_orbs import PitsAndOrbs

from wrappers.concat_observation import ConcatObservation
from wrappers.normalize_observation import NormalizeObservation
from wrappers.onehot_observation import OnehotObservation
from wrappers.steps_limit import StepsLimit

from utils import make_env


def run_env(env):
    print()
    print("---Sampled observation space from gym api:")
    print(env.observation_space.sample())
    print()

    obs = env.reset()
    print("---Sampled observation from reset function:")
    print(obs)
    print()

    for step in range(1000):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)

        if step % 100 == 0:
            print("Action:", action)
            print(obs)
            print("Reward:", reward)
            print()

        if done:
            break


if __name__ == "__main__":
    print("***Checking the raw game.")
    game = PitsAndOrbs(pygame_mode=True)
    game.reset_game()
    game.render_game(clear=False)
    game.close_game()
    del game

    print("***Checking the raw enviromnet without PyGame render.")
    env = PitsAndOrbsEnv()
    run_env(env)
    env.close()
    del env

    print("***Checking the raw enviromnet with PyGame render by sb3 check_env function.")
    env = PitsAndOrbsEnv(render_mode="human")
    check_env(env, skip_render_check=False)
    env.close()
    del env

    print("***Checking enviromnets made with gym.make with PyGame render by sb3 check_env function.")
    print(1)
    env = gym.make("EchineF/PitsAndOrbs-v0", render_mode="human")
    check_env(env, skip_render_check=False)
    env.close()
    del env

    print(2)
    env = gym.make("EchineF/PitsAndOrbs-two-players-v0", render_mode="human")
    check_env(env, skip_render_check=False)
    env.close()
    del env

    print("***Checking an enviromnet made with utils.make_env with PyGame render by sb3 check_env function.")
    env = make_env()
    check_env(env, skip_render_check=False)
    env.close()
    del env