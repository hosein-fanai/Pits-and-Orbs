import yaml

import time

from PIL import Image

import os

from utils import make_env


class Agent:

    def __init__(self, model_path=None):
        if model_path:
            self.load_model(model_path)

    def _run_rl_model_single_step(self, obs, deterministic=False, **kwargs):
        action, _ = self.model.predict(obs, deterministic=deterministic, **kwargs)

        return action

    def _run_classical(self, obs):
        return None

    def _agent_runner(self, env, max_steps=500, fps=None, 
                deterministic_action_choice=False, 
                print_rewards=True, return_frames=True):
        assert return_frames and env.game._pygame_mode

        obs = env.reset()
        done = False
        rewards = 0.

        frames = []
        frames_counter = 0

        start = time.time()
        for step in range(max_steps):
            action = self.predict(obs, deterministic=deterministic_action_choice)

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
        print("Total rewards were:", rewards)
        print("The episode was fps:", int(frames_counter/(time.time()-start)))
        print("The episode length was:", step+1)

        return rewards, frames

    @staticmethod
    def create_gif(frames, file_path):
        frame_images = [Image.fromarray(frame) for frame in frames]

        frame_images[0].save(
            file_path, format="GIF",
            append_images=frame_images,
            save_all=True,
            duration=90,
            loop=0
        )

    @staticmethod
    def load_train_configs(config_file_path):
        print("* Loading configs for training from", config_file_path)
        with open(config_file_path) as stream:
            configs = yaml.safe_load(stream)
        print('*')
        print("* Loaded configs from the file are:")
        print(configs)

        algorithm = list(configs.keys())[0].upper()
        configs = configs[algorithm]

        if algorithm == "DQN":
            iterations = configs["iterations"]
            iter_type = configs["iter_type"]
            n_step = configs["n_step"]
            target_soft_update = configs["target_soft_update"]
            target_update_interval = iterations * 0.15 * (0.5 if target_soft_update else 1)

            gamma = configs["gamma"] 
            buffer_size = configs["buffer_size"] 
            fill_buffer_episodes = configs["fill_buffer_episodes"]

            batch_size = configs["batch_size"]
            lr = configs["lr"]

            epsilon_active_portion = configs["epsilon_active_portion"]

            save_model_interval_portion = configs["save_model_interval_portion"]
            save_model_reward_threshold = configs["save_model_reward_threshold"]

            params = {
                "iterations": iterations,
                "iter_type": iter_type,
                "n_step": n_step,
                "soft_update": target_soft_update,
                "target_update_interval": target_update_interval,
                "gamma": gamma,
                "batch_size": batch_size,
                "warmup": fill_buffer_episodes,
                "epsilon_fn": configs["epsilon_fn"],
                "epsilon_active_portion": epsilon_active_portion,
                "save_model_interval_portion": save_model_interval_portion,
                "save_model_reward_threshold": save_model_reward_threshold,
            }

            kwargs = {
                "make_env": {**configs["make_env"]},
                "DQN": {"lr": lr, "buffer_size": buffer_size}
            }
        elif algorithm == "A2C":
            kwargs = {
                "n_env": configs["n_env"],
                "make_env": {**configs["make_env"]},
                "model_file_name": configs["model_file_name"],
                "verbose": configs["log_interval"],
                "log_interval": configs["log_interval"],
            }

            del configs["n_env"]
            del configs["make_env"]
            del configs["model_file_name"]
            del configs["verbose"]
            del configs["log_interval"]
            del configs["algorithm"]

            params = {**configs}
            
        return {"algorithm": algorithm, "params": params, "kwargs": kwargs}

    @staticmethod
    def load_run_configs(config_file_path):
        print("* Loading configs for running from", config_file_path)
        with open(config_file_path) as stream:
            configs = yaml.safe_load(stream)
        print('*')
        print("* Loaded configs from the file are:")
        print(configs)

        algorithm = list(configs.keys())[0].upper()
        configs = configs[algorithm]

        kwargs = {
            "make_env": {**configs["make_env"]},
            "model_file_name": configs["model_file_name"],
            "gif_file_name": configs["gif_file_name"],
        }

        del configs["make_env"]
        del configs["model_file_name"]
        del configs["gif_file_name"]

        params = {**configs}
        
        return {"algorithm": algorithm, "params": params, "kwargs": kwargs}

    def predict(self, obs, deterministic=False):
        if self.model:
            return self._run_rl_model_single_step(obs, deterministic)
        else:
            return self._run_classical(obs)

    def load_model(self, model_path):
        if model_path.endswith(".zip"):
            try:
                from stable_baselines3 import A2C
            except:
                print("Please install Stable-Baselines3 by: pip install stable-baselines3==1.8.0")
                return None

            self.model = A2C.load(model_path)
        elif model_path.endswith(".h5"):
            from agent.dqn import DQN

            self.model = DQN()
            self.load_model(model_path)

    def train_agent(self, config_file_path, do_save=True):
        configs = Agent.load_train_configs(config_file_path)

        if configs["algorithm"] == "DQN":
            from agent.dqn import DQN
            from agent.train_utils import plot_train_history


            env = make_env(**configs["kwargs"]["make_env"])
            self.model = DQN(env=env, **configs["kwargs"]["DQN"])

            rewards, total_loss = self.model.run_training(**configs["params"])
            plot_train_history(rewards, total_loss)
        elif configs["algorithm"] == "A2C":
            try:
                from stable_baselines3.common.env_util import make_vec_env
                from stable_baselines3 import A2C
            except:
                print("Please install Stable-Baselines3 by: pip install stable-baselines3==1.8.0")
                return None


            vec_env = make_vec_env(lambda: make_env(**configs["kwargs"]["make_env"]), n_envs=configs["kwargs"]["n_env"])
            self.model = A2C("MlpPolicy", vec_env,  **configs["params"], tensorboard_log=f"./logs/A2C")

            self.model.learn(total_timesteps=configs["kwargs"]["iterations"], log_interval=configs["kwargs"]["log_interval"])

        vec_env.close()
        if do_save:
            model_path = os.path.join("./models", configs["algorithm"], configs["kwargs"]["model_file_name"])
            self.model.save(model_path)

        return self.model

    def run_agent(self, config_file_path):
        configs = Agent.load_run_configs(config_file_path)

        model_path = os.path.join("./models", configs["algorithm"], configs["kwargs"]["model_file_name"])
        self.load_model(model_path)
        env = make_env(**configs["kwargs"]["make_env"])

        _, frames = self._agent_runner(env, **configs["params"])

        env.close()

        if configs["kwargs"].get("gif_file_name", None) is not None:
            gif_file_path = os.path.join("./gifs", configs["kwargs"]["gif_file_name"])
            Agent.create_gif(frames, gif_file_path)

            print()
            print(".gif file saved to:", gif_file_path)
        else:
            print()
            print("Skipped creating .gif file since no path was provided to save it.")