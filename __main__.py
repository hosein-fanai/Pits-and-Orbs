from argparse import ArgumentParser

from agent.agent import Agent


parser = ArgumentParser(description="A multi-agent-system project (called Pits and Orbs) which is able to be ran via cli.")
parser.add_argument("--file", help="The config file path for previuos flag.")
parser.add_argument("--run", "-r", action="store_true", help="The argument to run the agent on the environment \
                    with the given config file (can't be used with train flag on).")
parser.add_argument("--train", "-t", action="store_true", help="The argument to train the agent with the specified \
                    with the given config file(can't be used with run flag on).")


args = parser.parse_args()
config_file_path = args.file
run_flag = args.run
train_flag = args.train

assert not(run_flag and train_flag) and (run_flag or train_flag)

if run_flag:
    agent = Agent()
    agent.run_agent(config_file_path=config_file_path)
elif train_flag:
    agent = Agent()
    agent.train_agent(config_file_path=config_file_path)