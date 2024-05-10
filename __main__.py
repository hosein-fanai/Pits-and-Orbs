from argparse import ArgumentParser

from rl.agent import Agent


parser = ArgumentParser(description="A multi-agent-system project (called Pits and Orbs) which is able to be used via cli.")
parser.add_argument("--run", "-r", action="store_true", help="The argument to run the agent on the environment \
                    with the given config file (can't be used with train flag on).")
parser.add_argument("--train", "-t", action="store_true", help="The argument to train the agent with the specified \
                    with the given config file(can't be used with run flag on).")
parser.add_argument("config", help="The config file path for previuos flag.")

args = parser.parse_args()
run_flag = args.run
train_flag = args.train
config_file_path = args.config

assert not(run_flag and train_flag) and (run_flag or train_flag)

agent = Agent()

if run_flag:
    agent.run_agent(config_file_path=config_file_path)
elif train_flag:
    agent.train_agent(config_file_path=config_file_path)