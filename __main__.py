from argparse import ArgumentParser

import yaml

from utils import make_env, run_agent, create_gif

from agent.agent import Agent


parser = ArgumentParser(description="A multi-agent-system project (called Pits and Orbs) which is able to be ran via cli.")
parser.add_argument("--run", "-r", action="store_true", help="The argument to run the agent on the environment \
                    with the given config file (can't be used with train flag on).")
parser.add_argument("--train", "-t", action="store_true", help="The argument to train the agent with the specified \
                    with the given config file(can't be used with run flag on).")

args = parser.parse_args()
run_arg = args.run
train_arg = args.train

assert not(run_arg and train_arg) and (run_arg or train_arg)

if run_arg:
    agent = Agent()
    agent.run_agent(config_file_path=run_arg)

elif train_arg:
    agent = Agent()
    agent.train_agent(config_file_path=train_arg)