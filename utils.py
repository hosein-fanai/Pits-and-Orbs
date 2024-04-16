from gym import wrappers

from PIL import Image

import time

from environment.pits_and_orbs_env import PitsAndOrbsEnv

from wrappers.onehot_observation import OnehotObservation
from wrappers.normalize_observation import NormalizeObservation


def run_agent(env, agent, max_steps=1000, print_rewards=True, return_frames=True):
    assert return_frames and env.game._pygame_mode

    obs = env.reset()
    done = False
    rewards = 0

    frames = []
    frames_counter = 0
    
    start = time.time()
    for _ in range(max_steps):
        action = agent.predict(obs)

        obs, reward, done, _ = env.step(action)

        rewards += reward
        frames_counter += 1

        if print_rewards:
            print(f"\rTotal Rewards: {rewards}, Current Reward: {reward}", end="")

        if return_frames:
            frame = env.render("rgb_array")
            frames.append(frame)

        if done:
            break

    print(f"1 Episode is done with the fps of {int((time.time() - start)/frames_counter)}")

    return rewards, frames


def create_gif(frames, file_path):
    frame_images = [Image.fromarray(frame) for frame in frames]

    frame_images[0].save(
        file_path, format='GIF',
        append_images=frame_images[1:],
        save_all=True,
        duration=30,
        loop=0
    )


def make_env(pygame_mode=False, max_movements=30, size=(5, 5), 
            orb_num=5, pit_num=5, seed=None, onehot_obs=True, 
            norm_obs=False, num_stack=None):
    env = PitsAndOrbsEnv(pygame_mode=pygame_mode, max_movements=max_movements, 
                        size=size, orb_num=orb_num, pit_num=pit_num, seed=seed)

    if onehot_obs and not norm_obs:
        env = OnehotObservation(env)

    if norm_obs and not onehot_obs:
        env = NormalizeObservation(env)

    if num_stack:
        env = wrappers.FrameStack(env, num_stack=num_stack)

    return env