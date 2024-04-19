from argparse import ArgumentParser

import yaml

from utils import make_env, run_agent, create_gif

from agent.agent import Agent


parser = ArgumentParser(description="A multi-agents-system project (called Pits and Orbs) which is able to be ran via cli.")
parser.add_argument("--run", "-r", action="store_true", help="The flag to run the agent on the environment (can't be used with train flag on).")
parser.add_argument("--train", "-t", action="store_true", help="The flag to train the agent with the specified ./agent/config.yaml file (can't be used with run flag on).")
parser.add_argument("--mpath", help="The file path to an RL model.")
parser.add_argument("--gpath", help="The save file path for gif file containing an episode's frames.")

args = parser.parse_args()
run_flag = args.run
train_flag = args.train

assert not(run_flag and train_flag) and (run_flag or train_flag)

if run_flag:
    model_path = args.mpath
    gif_path = args.gpath

    env = make_env(render_mode="human")
    agent = Agent(env, model_path)

    print()
    print("Using the agent located at:", model_path)
    print()

    rewards, frames = run_agent(env, agent, return_frames=True)

    if gif_path is not None:
        create_gif(frames, gif_path)
        print()
        print(".gif file saved to:", gif_path)
    else:
        print()
        print("Skipped creating .gif file since no path was provided to save it.")
elif train_flag:
    with open("./agent/config.yaml") as stream:
        configs = yaml.safe_load(stream)

        print("* Initializing constants of the project according to ./agent/config.yaml")
        print('*')
        print("* Loaded configs from the file are:")
        print(configs)

    lr = int(["lr"])