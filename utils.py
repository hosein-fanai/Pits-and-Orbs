from gym import wrappers

from PIL import Image

import time

from environment.pits_and_orbs_env import PitsAndOrbsEnv

from wrappers.onehot_observation import OnehotObservation
from wrappers.normalize_observation import NormalizeObservation
from wrappers.concat_observation import ConcatObservation


def run_agent(env, agent, max_steps=1000, fps=None, 
            deterministic_action_choice=False, 
            print_rewards=True, return_frames=True):
    assert return_frames and env.game._pygame_mode

    obs = env.reset()
    done = False
    rewards = 0

    frames = []
    frames_counter = 0
    
    start = time.time()
    for step in range(max_steps):
        action = agent.predict(obs, deterministic=deterministic_action_choice)

        obs, reward, done, info = env.step(action)

        rewards += reward
        frames_counter += 1

        if print_rewards:
            print(f"\rTotal Rewards: {rewards}, Current Reward: {reward}", end="")

        if return_frames:
            frame = env.render("rgb_array")
            frames.append(frame)

        if done:
            print()
            print("One episode is done successfully with ", end="")
            for i in range(env.game.player_num):
                print(f"player{i} having done {info[f'player{i} movements#']} movements ", end="")
            print()
            break

        if fps is not None:
            env.game.clock.tick(fps)

    print()
    print("The episode was fps:", int(frames_counter/(time.time()-start)))
    print("The episode length was:", step+1)

    return rewards, frames


def create_gif(frames, file_path):
    frame_images = [Image.fromarray(frame) for frame in frames]

    frame_images[0].save(
        file_path, format='GIF',
        append_images=frame_images,
        save_all=True,
        duration=60,
        loop=0
    )


def make_env(render_mode="rgb_array", max_movements=30, return_partial_obs=True,
            size=(5, 5), orb_num=5, pit_num=5, player_num=1, seed=None, 
            onehot_obs=True, norm_obs=False, num_stack=None):
    env = PitsAndOrbsEnv(render_mode=render_mode, pygame_with_help=False, 
                        return_partial_obs=return_partial_obs, max_movements=max_movements, 
                        size=size, orb_num=orb_num, pit_num=pit_num, player_num=player_num, 
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
    env = make_env()
    print(env.observation_space.shape)

    obs = env.reset()
    print(obs)
    print(obs.shape)