from argparse import ArgumentParser

from utils import make_env, run_agent, create_gif

from agent.agent import Agent


parser = ArgumentParser(description="A multi-agents-system project (called Pits and Orbs) which is able to be ran via cli.")
parser.add_argument("mpath", help="The file path to an RL model.")
parser.add_argument("gpath", help="The save file path for gif file containing an episode's frames.")

args = parser.parse_args()
model_path = args.mpath
gif_path = args.gpath

env = make_env(num_stack=4, pygame_mode=True)
agent = Agent(env, model_path)

print()
print("Using the agent located at:", model_path)
print()

rewards, frames = run_agent(env, agent, return_frames=True)

create_gif(frames, gif_path)
print()
print("Gif file saved to:", gif_path)